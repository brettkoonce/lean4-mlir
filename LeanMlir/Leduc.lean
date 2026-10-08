import LeanMlir.FloatFmt
import LeanMlir.F32Array

/-! Leduc hold'em: the game, the exact instrument and Deep CFR's samplers. The game is pure
    Lean; every tree walk is in C (`ffi/f32_helpers.c`, `lean_leduc_*`), as tic-tac-toe's
    solver is. Shared by the `leduc-env` and `deep-cfr-leduc` demos.

    Rules as OpenSpiel's `leduc_poker` spells them (Southey et al. 2005, *Bayes' Bluff*): 2r
    cards, r ranks × 2 suits (r = 3 is Leduc: J Q K), ante 1 each, one private card each, two
    betting rounds with bets of 2 then 4 and at most two raises a round, fold legal only when
    facing a bet, P0 acts first in both rounds, one public card between the rounds; at
    showdown a pair with the public card wins, else the higher rank, equal ranks split.

    Suits never matter (no flushes), so the instrument works at RANK level: an information set
    is (round-1 betting state, private rank) in round 1 and (round-1 closing, round-2 state,
    private rank, public rank) in round 2 — 6r + 30r² of them, 288 at r = 3 against
    OpenSpiel's 936 suit-aware sets, which fold onto ours one-to-one. A strategy is a dense f32
    `[nInfo, 3]` table over fold / call / raise, illegal slots ignored; `exploitability` is the
    best responder's winnings per hand, averaged over the two seats (NashConv / 2, OpenSpiel's
    convention), and zero is Nash. -/

namespace Leduc

open FloatFmt

inductive Action | fold | call | raise
deriving Inhabited, Repr, BEq, DecidableEq

def Action.idx : Action → Nat | .fold => 0 | .call => 1 | .raise => 2
def Action.ofIdx : Nat → Action | 0 => .fold | 1 => .call | _ => .raise
def Action.str : Action → String | .fold => "f" | .call => "c" | .raise => "r"

inductive Terminal | fold (who : Nat) | showdown
deriving Inhabited, Repr

/-- A hand in play: concrete cards (`0 .. 2r − 1`, rank = card / 2, as OpenSpiel numbers
    them), the two rounds' action strings over `c` / `r`, the contributions to the pot. -/
structure State where
  r : Nat
  cards : Array Nat          -- P0's and P1's private cards
  pub : Option Nat := none    -- the public card, dealt between the rounds
  round : Nat := 0
  seqs : Array String := #["", ""]
  contrib : Array Nat := #[1, 1]
  player : Nat := 0
  needPub : Bool := false     -- round 1 closed, the public card is next
  term : Option Terminal := none
deriving Inhabited, Repr

def rank (card : Nat) : Nat := card / 2

def State.rankOf (s : State) (p : Nat) : Nat := rank s.cards[p]!
def State.pubRank (s : State) : Option Nat := s.pub.map rank

def State.facing (s : State) : Bool := s.contrib[s.player]! < max s.contrib[0]! s.contrib[1]!
def State.nRaises (s : State) : Nat := (s.seqs[s.round]!.toList.filter (· == 'r')).length

def State.legal (s : State) : Array Action := Id.run do
  let mut a : Array Action := #[]
  if s.facing then a := a.push .fold
  a := a.push .call
  if s.nRaises < 2 then a := a.push .raise
  return a

/-- One action; the caller deals the public card (`dealPub`) when `needPub` comes up. -/
def State.step (s : State) (a : Action) : State := Id.run do
  let p := s.player
  let size := if s.round == 0 then 2 else 4
  let mx := max s.contrib[0]! s.contrib[1]!
  match a with
  | .fold => return { s with term := some (.fold p) }
  | .call =>
    let s := { s with contrib := s.contrib.set! p mx, seqs := s.seqs.modify s.round (· ++ "c") }
    let closed := s.seqs[s.round]!.length >= 2
    if closed then
      if s.round == 0 then return { s with round := 1, player := 0, needPub := true }
      else return { s with term := some .showdown }
    return { s with player := 1 - p }
  | .raise =>
    return { s with contrib := s.contrib.set! p (mx + size),
                    seqs := s.seqs.modify s.round (· ++ "r"), player := 1 - p }

def State.dealPub (s : State) (card : Nat) : State := { s with pub := some card, needPub := false }

/-- The payoff to P0 of a finished hand. -/
def State.payoff (s : State) : Float :=
  match s.term with
  | some (.fold 0) => -(s.contrib[0]!).toFloat
  | some (.fold _) => (s.contrib[1]!).toFloat
  | some .showdown =>
    let pub := s.pubRank.getD 1000
    let strength (c : Nat) := if rank c == pub then 100 + rank c else rank c
    let s0 := strength s.cards[0]!
    let s1 := strength s.cards[1]!
    if s0 > s1 then (s.contrib[1]!).toFloat else if s1 > s0 then -(s.contrib[0]!).toFloat else 0.0
  | none => 0.0

/-- The information-set key of the player to move: their private rank, the public rank once
    dealt, and the two rounds' action strings. -/
def State.infoKey (s : State) : Nat × Nat × Option Nat × String × String :=
  (s.player, s.rankOf s.player, s.pubRank, s.seqs[0]!, s.seqs[1]!)

/-- The betting-round automaton's states, as the C instrument numbers them. -/
def stateIndex (seq : String) : Nat :=
  match seq with | "" => 0 | "c" => 1 | "r" => 2 | "cr" => 3 | "rr" => 4 | _ => 5
def closingIndex (seq : String) : Nat :=
  match seq with | "cc" => 0 | "rc" => 1 | "crc" => 2 | "rrc" => 3 | _ => 4

/-- The dense index of the player to move's information set (C's `ld_info_index`). -/
def State.infoIndex (s : State) : Nat :=
  let r := s.r
  let a := s.rankOf s.player
  if s.round == 0 then stateIndex s.seqs[0]! * r + a
  else 6 * r + (((closingIndex s.seqs[0]! * 6 + stateIndex s.seqs[1]!) * r + a) * r + s.pubRank.getD 0)

def nInfo (r : Nat) : Nat := 6 * r + 30 * r * r
def nFeatures (r : Nat) : Nat := 2 * r + 24

/-- Two distinct cards, uniformly. -/
def deal (r : Nat) (g : StdGen) : State × StdGen :=
  let (c0, g) := randNat g 0 (2 * r - 1)
  let (c1', g) := randNat g 0 (2 * r - 2)
  let c1 := if c1' >= c0 then c1' + 1 else c1'
  ({ r, cards := #[c0, c1] }, g)

/-- A public card among those not held, uniformly. -/
def drawPub (s : State) (g : StdGen) : State × StdGen :=
  let left := (List.range (2 * s.r)).filter (fun c => !s.cards.contains c)
  let (i, g) := randNat g 0 (left.length - 1)
  (s.dealPub left[i]!, g)

def cardName (r : Nat) (card : Nat) : String :=
  let names := if r == 3 then #["J", "Q", "K"] else (Array.range r).map fun i => s!"{i}"
  names[rank card]! ++ (if card % 2 == 0 then "♠" else "♥")

def State.render (s : State) : String :=
  s!"P0 {cardName s.r s.cards[0]!} P1 {cardName s.r s.cards[1]!}" ++
  (match s.pub with | some c => s!" public {cardName s.r c}" | none => "") ++
  s!" round {s.round + 1} [{s.seqs[0]!}|{s.seqs[1]!}] pot {s.contrib[0]!}+{s.contrib[1]!}"

/-- Every terminal history of the Lean game in the C instrument's canonical order (P0's card,
    P1's card, actions fold < call < raise depth-first, public cards ascending). -/
partial def allTerminals (r : Nat) : Array State := Id.run do
  let rec walk (s : State) (acc : Array State) : Array State := Id.run do
    let mut acc := acc
    if s.term.isSome then return acc.push s
    if s.needPub then
      for c in [0:2 * r] do
        if !s.cards.contains c then acc := walk (s.dealPub c) acc
      return acc
    for a in s.legal do acc := walk (s.step a) acc
    return acc
  let mut acc := #[]
  for c0 in [0:2 * r] do
    for c1 in [0:2 * r] do
      if c0 != c1 then acc := walk { r, cards := #[c0, c1] } acc
  return acc

/-- Every terminal payoff in that order — compared against `payoffs` by `leduc-env`'s Gate 0. -/
def allPayoffs (r : Nat) : Array Float := (allTerminals r).map State.payoff

/-- A finished hand as one line of the OpenSpiel cross-check
    (`scripts/demos/leduc_gate0_openspiel.py`): the concrete cards (the public card −1 when
    the hand ended in round 1), the actions as OpenSpiel numbers them (0 fold, 1 call,
    2 raise), the payoff to P0. -/
def State.historyLine (s : State) : String :=
  let acts := s.seqs[0]!.toList ++ (if s.round == 1 then s.seqs[1]!.toList else [])
  let acts := acts.map fun ch => if ch == 'c' then "1" else "2"
  let acts := match s.term with | some (.fold _) => acts ++ ["0"] | _ => acts
  let pub := match s.pub with | some c => s!"{c}" | none => "-1"
  s!"{s.cards[0]!} {s.cards[1]!} {pub} {" ".intercalate acts} : {s.payoff}"

/-- A strategy's probabilities at a state, read from a table. -/
def tableAt (tbl : ByteArray) (s : State) : Array Float :=
  (Array.range 3).map fun x => F32.read tbl (s.infoIndex * 3 + x).toUSize

-- ── The C instrument ──

/-- u64 `[information sets, F, suit-aware history nodes]`. -/
@[extern "lean_leduc_counts"]
opaque counts (r : USize) : IO ByteArray

/-- Every information set's feature row and legal mask: `(f32 [nInfo, F], f32 [nInfo, 3])`. -/
@[extern "lean_leduc_enumerate"]
opaque enumerate (r : USize) : IO (ByteArray × ByteArray)

/-- Every information set's key: u8 `[nInfo, 6]` = player, round, round-1 closing (255 in
    round 1), betting state, private rank, public rank (255 in round 1). -/
@[extern "lean_leduc_keys"]
opaque keys (r : USize) : IO ByteArray

/-- Every terminal history's payoff to P0 in the canonical order of `allPayoffs`. -/
@[extern "lean_leduc_payoffs"]
opaque payoffs (r : USize) : IO ByteArray

@[extern "lean_leduc_uniform"]
opaque uniformTable (r : USize) : IO ByteArray

/-- The scripted "honest" arm: raise with a pair or the top rank, fold the bottom rank to a
    bet, call otherwise. -/
@[extern "lean_leduc_scripted_honest"]
opaque honestTable (r : USize) : IO ByteArray

/-- The same profile with its bluffs removed: the worst hand's raise mass moves to call. -/
@[extern "lean_leduc_remove_bluffs"]
opaque removeBluffs (r : USize) (table : @& ByteArray) : IO ByteArray

/-- u8 `[nInfo]`: 1 where the set holds the worst hand (a holding that beats nothing a
    showdown can produce except a tie). -/
@[extern "lean_leduc_worst_hands"]
opaque worstHands (r : USize) : IO ByteArray

/-- f32 `[exploitability, best-response value as P0, as P1]` — exact, one walk of the tree. -/
@[extern "lean_leduc_exploitability"]
opaque exploitabilityOf (r : USize) (table : @& ByteArray) : IO ByteArray

/-- The pure best response to a table, both seats, as a table. -/
@[extern "lean_leduc_best_response"]
opaque bestResponse (r : USize) (table : @& ByteArray) : IO ByteArray

/-- Expected payoff to P0 when `a` plays P0's rows and `b` plays P1's — exact. -/
@[extern "lean_leduc_head_to_head"]
opaque headToHead (r : USize) (a b : @& ByteArray) : Float

/-- Head-to-head of `a` against `b` averaged over the two seats, from `a`'s side. -/
def seatAveraged (r : Nat) (a b : ByteArray) : Float :=
  0.5 * (headToHead r.toUSize a b - headToHead r.toUSize b a)

-- ── The tabular solvers ──

/-- A solver arena: `mode` 0 CFR+ (regret matching+, linear averaging), 1 DCFR (Brown &
    Sandholm 2019), 2 vanilla; `eps > 0` starts from random positive regrets (the
    non-uniqueness check runs two seeds per solver). -/
@[extern "lean_leduc_cfr_alloc"]
opaque cfrAlloc (r : USize) (mode : UInt8) (seed : UInt64) (eps : Float) : IO ByteArray

@[extern "lean_leduc_cfr_iterate"]
opaque cfrIterate (arena : @& ByteArray) (iters : USize) : IO Unit

/-- The average strategy — the profile that converges. -/
@[extern "lean_leduc_cfr_average"]
opaque cfrAverage (arena : @& ByteArray) : IO ByteArray

/-- The current profile (regret matching on the accumulated regrets). -/
@[extern "lean_leduc_cfr_current"]
opaque cfrCurrent (arena : @& ByteArray) : IO ByteArray

/-- External-sampling MCCFR (Lanctot et al. 2009): the matched-budget tabular bracket. -/
@[extern "lean_leduc_es_alloc"]
opaque esAlloc (r : USize) (seed : UInt64) : IO ByteArray

/-- Run until `budget` nodes have been touched in all: u64 `[nodes, iterations, sets visited]`. -/
@[extern "lean_leduc_es_run"]
opaque esRun (arena : @& ByteArray) (budget : UInt64) : IO ByteArray

@[extern "lean_leduc_es_average"]
opaque esAverage (arena : @& ByteArray) : IO ByteArray

-- ── Deep CFR's reservoirs and traversal ──

/-- A reservoir of `cap` rows `[features F, target 3, mask 3, t, set index]`. -/
@[extern "lean_leduc_reservoir_alloc"]
opaque reservoirAlloc (F : USize) (cap : UInt64) : IO ByteArray

@[extern "lean_leduc_reservoir_reset"]
opaque reservoirReset (res : @& ByteArray) : IO Unit

/-- u64 `[rows held, rows seen]`. -/
@[extern "lean_leduc_reservoir_stats"]
opaque reservoirStats (res : @& ByteArray) : IO ByteArray

/-- `K` external-sampling traversals for traverser `p` at iteration `t` against the tables
    `sig0` / `sig1` (the nets' regret-matched advantages over every set): instantaneous
    regrets into `adv`, the opponent's σ into `strat`. Returns the nodes touched. -/
@[extern "lean_leduc_traverse"]
opaque traverse (r : USize) (sig0 sig1 feats mask : @& ByteArray) (p K : USize) (t : Float)
    (seed : UInt64) (adv strat : @& ByteArray) : IO UInt64

/-- A batch with replacement: `(x f32 [B, F], tail f32 [B, 8] = target 3, mask 3, t, index)`. -/
@[extern "lean_leduc_reservoir_sample"]
opaque reservoirSample (res : @& ByteArray) (B : USize) (seed : UInt64) : IO (ByteArray × ByteArray)

/-- Gate B's table in the net's place: the summed sampled regrets of an advantage reservoir
    per set, regret-matched — ES-MCCFR's current strategy. Every row must still be held. -/
@[extern "lean_leduc_reservoir_sigma"]
opaque reservoirSigma (r : USize) (res : @& ByteArray) : IO ByteArray

/-- The t-weighted average of a strategy reservoir's σ rows per set — ES-MCCFR's average
    strategy when every row is held. -/
@[extern "lean_leduc_reservoir_average"]
opaque reservoirAverage (r : USize) (res : @& ByteArray) : IO ByteArray

/-- The MSE block's target for the weighted, masked squared error on a logits block `[B, 3]`:
    `y = out − scale · g`, `g = w ⊙ m ⊙ (out − target)`, `w = t / mean t`; at `scale = nOut / 2`
    the block's gradient is `g / B`. Returns `(y [B, 3], [mean weighted loss])`. -/
@[extern "lean_leduc_targets"]
opaque targets (out tail : @& ByteArray) (B : USize) (scale : Float) : IO (ByteArray × ByteArray)

/-- σ from an advantage net's output over every set: regret matching on the positive part,
    the argmax when none is positive (Brown et al. 2019). -/
@[extern "lean_leduc_sigma_from_advantages"]
opaque sigmaFromAdvantages (r : USize) (adv : @& ByteArray) : IO ByteArray

/-- σ from the strategy net's output: the positive part normalised over the legal slots. -/
@[extern "lean_leduc_sigma_from_strategy_net"]
opaque sigmaFromStrategyNet (r : USize) (out : @& ByteArray) : IO ByteArray

/-- SD-CFR (Steinberger 2019): the exact average of `T` stored profiles `[T, nInfo, 3]` with
    weights `[T]` (the iteration numbers), each weighted by its own reach of the set. -/
@[extern "lean_leduc_sdcfr_average"]
opaque sdcfrAverage (r : USize) (tables weights : @& ByteArray) (T : USize) : IO ByteArray

/-- u64 record `i` of a packed buffer. -/
def readU64 (ba : ByteArray) (i : Nat) : Nat := readU64LE ba (8 * i)

/-- Exploitability alone. -/
def exploitability (r : Nat) (table : ByteArray) : IO Float := do
  let e ← exploitabilityOf r.toUSize table
  return (F32.read e 0)

end Leduc
