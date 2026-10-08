#!/usr/bin/env python3
"""Gate 0 of the Leduc demo — planning/leduc_deep_cfr_demo.md §7: the Lean game and the C
instrument against OpenSpiel's `leduc_poker`, on the files `lake exe leduc-env` writes to
.lake/build/.

1. Every terminal history of the Lean game (`leduc_r3_payoffs.txt`: concrete cards 0..5,
   rank = card // 2, the actions as OpenSpiel numbers them, the payoff to P0) has OpenSpiel's
   `returns()[0]` on the same history — 5,520 of them.
2. The CFR+ and DCFR tables (`leduc_r3_cfrplus.txt`, `leduc_r3_dcfr.txt`: 288 rank-level rows),
   lifted to OpenSpiel's 936 information states one-to-one, have the exploitability OpenSpiel
   computes for them, within 1e-6 of what the C instrument printed (`leduc_r3_gate0.txt`); the
   uniform policy likewise. Exit 1 on any failure.

  .venv-poker/bin/python scripts/demos/leduc_gate0_openspiel.py [.lake/build]

The venv: `python3.12 -m venv .venv-poker && .venv-poker/bin/pip install -r
requirements-poker-lock.txt` — never the main .venv (open_spiel pulls its own
numpy / scipy; see scripts/gates/check_pinned_env.py for why that matters).
"""
import sys
import numpy as np
import pyspiel
from open_spiel.python import policy as pl
from open_spiel.python.algorithms import exploitability

OUT = sys.argv[1] if len(sys.argv) > 1 else ".lake/build"
g = pyspiel.load_game("leduc_poker")
assert g.num_players() == 2 and g.num_distinct_actions() == 3

# ---- OpenSpiel's card → rank: test both conventions against the payoff dump ----
ours = {}
with open(f"{OUT}/leduc_r3_payoffs.txt") as f:
    for line in f:
        head, pay = line.split(":")
        v = [int(t) for t in head.split()]
        ours[tuple(v)] = float(pay)
print("our terminal histories:", len(ours))

def walk_terminals(state, cards, acts, out):
    if state.is_terminal():
        out.append((tuple(cards), tuple(acts), state.returns()[0]))
        return
    if state.is_chance_node():
        for a, _ in state.chance_outcomes():
            walk_terminals(state.child(a), cards + [a], acts, out)
        return
    for a in state.legal_actions():
        walk_terminals(state.child(a), cards, acts + [a], out)

terms = []
walk_terminals(g.new_initial_state(), [], [], terms)
print("OpenSpiel terminal histories:", len(terms))

def check_map(card_map):
    bad = 0
    for cards, acts, ret in terms:
        c0, c1 = card_map(cards[0]), card_map(cards[1])
        pub = card_map(cards[2]) if len(cards) == 3 else -1
        key = (c0, c1, pub) + acts
        if key not in ours or abs(ours[key] - ret) > 1e-9:
            bad += 1
    return bad

maps = {"rank = card // 2 (suit = card % 2)": lambda c: c,
        "rank = card % 3 (suit = card // 3)": lambda c: (c % 3) * 2 + c // 3}
chosen = None
for name, m in maps.items():
    bad = check_map(m)
    print(f"  payoff gate under {name}: {bad} mismatches of {len(terms)}")
    if bad == 0:
        chosen = m
assert chosen is not None, "no card convention reproduces OpenSpiel's payoffs"
assert len(ours) == len(terms)

# ---- lift a rank-level table to OpenSpiel's information states ----
STATES = {"": 0, "c": 1, "r": 2, "cr": 3, "rr": 4, "crr": 5}
CLOSINGS = {"cc": 0, "rc": 1, "crc": 2, "rrc": 3, "crrc": 4}
R = 3
def info_index(round_, closing, s, a, pub):
    if round_ == 0:
        return s * R + a
    return 6 * R + (((closing * 6 + s) * R + a) * R + pub)

def load_table(path):
    tab = np.zeros((6 * R + 30 * R * R, 3))
    with open(path) as f:
        for line in f:
            v = line.split()
            i = int(v[0]); tab[i] = [float(v[6]), float(v[7]), float(v[8])]
    # the rows are f32 at the Lean boundary; renormalise the ~1e-7 the rounding leaves
    return tab / np.maximum(tab.sum(axis=1, keepdims=True), 1e-12)

def key_of(state, cards, acts):
    """our information-set index for the player to move at `state`"""
    p = state.current_player()
    a = chosen(cards[p]) // 2
    # split the public action list into rounds: round 1 ends at a closing
    r1, r2 = [], []
    seq = ""
    closed = False
    for x in acts:
        ch = "c" if x == 1 else "r"
        if not closed:
            r1.append(ch); seq += ch
            if ch == "c" and len(seq) >= 2:
                closed = True
        else:
            r2.append(ch)
    s1 = "".join(r1); s2 = "".join(r2)
    if not closed:
        return info_index(0, -1, STATES[s1], a, -1)
    pub = chosen(cards[2]) // 2
    return info_index(1, CLOSINGS[s1], STATES[s2], a, pub)

def lift(tab):
    pol = pl.TabularPolicy(g)
    seen = {}
    def walk(state, cards, acts):
        if state.is_terminal():
            return
        if state.is_chance_node():
            for a, _ in state.chance_outcomes():
                walk(state.child(a), cards + [a], acts)
            return
        k = state.information_state_string()
        i = key_of(state, cards, acts)
        if k in seen:
            assert seen[k] == i, (k, seen[k], i)
        else:
            seen[k] = i
            row = pol.policy_for_key(k)
            legal = state.legal_actions()
            for x in range(3):
                row[x] = tab[i, x] if x in legal else 0.0
            assert abs(sum(row) - 1) < 1e-6, (k, row)
        for a in state.legal_actions():
            walk(state.child(a), cards, acts + [a])
    walk(g.new_initial_state(), [], [])
    print(f"  {len(seen)} OpenSpiel information states lifted from {len(set(seen.values()))} rank-level sets")
    return pol

ours_e = {}
with open(f"{OUT}/leduc_r3_gate0.txt") as f:
    for line in f:
        k, v = line.split()
        ours_e[k] = float(v)

TOL = 1e-6
bad = 0
u = pl.UniformRandomPolicy(g)
e = exploitability.exploitability(g, u)
print(f"uniform: OpenSpiel exploitability {e:.9f}, ours {ours_e['uniform']:.9f}")
bad += abs(e - ours_e["uniform"]) > TOL
for name in ("cfrplus", "dcfr"):
    tab = load_table(f"{OUT}/leduc_r3_{name}.txt")
    pol = lift(tab)
    e = exploitability.exploitability(g, pol)
    nc = exploitability.nash_conv(g, pol)
    print(f"{name}: OpenSpiel exploitability {e:.9e} (nash_conv {nc:.9e}), ours {ours_e[name]:.9e}")
    bad += abs(e - ours_e[name]) > TOL
if bad:
    sys.exit(f"gate 0 FAILED: {bad} exploitability reading(s) differ from OpenSpiel's by more than {TOL}")
print("gate 0: payoffs identical, tables lift one-to-one, exploitabilities agree with OpenSpiel's to 1e-6")
