"""Size a candidate retirement group against the use graph.

seed: pins (and named unpinned decls) to cut. Unpinned declarations whose every user is in the
cut set are followed to a fixed point; pins outside the seed that lose their last non-audit user
are reported (not cut) — those are a decision. Names in `keep` are never cut (they count as users).
Already-dead unpinned decls are not counted.
"""
import json, sys, pickle, os
from collections import defaultdict
from pathlib import Path
G = pickle.load(open(Path(os.environ.get("CENSUS_DIR", "/tmp/audit_census")) / "graph.pkl", "rb"))
consts, env_users, tok_users, pins = G["consts"], G["env_users"], G["tok_users"], G["pins"]
pinset = {p for p, _ in pins}
decls = {n for n, c in consts.items() if c["lo"] is not None}
U = {}
for n in decls:
    s = set(env_users.get(n, ()))
    for t in tok_users.get(n, ()):
        s.add(t[len("mention:"):] if t.startswith("mention:") else "EXT:" + t)
    U[n] = s
USES = defaultdict(set)
for n, us in U.items():
    for u in us:
        if not u.startswith("EXT:"): USES[u].add(n)

def lines(n): c = consts[n]; return c["hi"] - c["lo"] + 1

def resolve(name):
    if name in consts: return name
    for pre in ("Proofs.", "Proofs.StableHLO.", "Proofs.IR."):
        if pre + name in consts: return pre + name
    hits = [n for n in decls if n.endswith("." + name)]
    if len(hits) == 1: return hits[0]
    raise KeyError(f"{name}: {hits[:5]}")

def size(seed_names, follow_pins=(), keep=()):
    keep = {resolve(k) for k in keep}
    seed = {resolve(s) for s in seed_names}
    extra_pins = {resolve(s) for s in follow_pins}
    R = set(seed) | extra_pins
    orphan_pins = set()
    changed = True
    while changed:
        changed = False
        cand = set()
        for r in R: cand |= USES.get(r, set())
        for d in cand:
            if d in R or d in keep: continue
            us = U[d]
            if us and us <= R:
                if d in pinset:
                    orphan_pins.add(d)
                else:
                    R.add(d); changed = True
    orphan_pins -= R
    pins_cut = sorted(x for x in R if x in pinset)
    return dict(decls=len(R), lines=sum(lines(x) for x in R), pins=len(pins_cut),
                pins_cut=pins_cut, unpinned_cut=sorted(x for x in R if x not in pinset),
                newly_audit_only_pins=sorted(orphan_pins),
                files=sorted({consts[x]["file"] for x in R}))

if __name__ == "__main__":
    groups = json.load(open(sys.argv[1]))
    out = {}
    for g, spec in groups.items():
        r = size(spec["seed"], spec.get("follow", []), spec.get("keep", []))
        out[g] = r
        print(f"{g:40s} decls {r['decls']:4d}  lines {r['lines']:5d}  pins {r['pins']:3d}  "
              f"+{len(r['newly_audit_only_pins'])} pins would lose their last user")
    json.dump(out, open(sys.argv[2], "w"), indent=1)
