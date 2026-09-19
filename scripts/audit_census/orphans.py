"""    python3 scripts/audit_census/orphans.py groups.json <census dir>

Env-graph-only orphan pass: unpinned declarations whose every ELABORATED user is in the cut set
(token users ignored), iterated. Candidates only — vet each against #guard/test/exe use.
Needed because a common short name (`X`, `fwd`, `stem`) collides with tokens elsewhere, so the
union graph size.py uses stops the fixed point early."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["CENSUS_DIR"] = sys.argv[2]
import size as S
g = json.load(open(sys.argv[1]))
seed, follow, keep = [], [], []
for v in g.values(): seed += v["seed"]; follow += v.get("follow", []); keep += v.get("keep", [])
r = S.size(seed, follow, keep)
cut = set(r["pins_cut"] + r["unpinned_cut"])
envU = {n: set(S.env_users.get(n, ())) for n in S.decls}
R = set(cut); new = []
changed = True
while changed:
    changed = False
    for d in S.decls:
        if d in R or d in S.pinset: continue
        u = envU[d]
        if u and u <= R:
            R.add(d); new.append(d); changed = True
for d in sorted(new, key=lambda n: (S.consts[n]["file"], S.consts[n]["lo"])):
    c = S.consts[d]
    toks = sorted(t for t in S.tok_users.get(d, ()) if not t.startswith("mention:"))[:3]
    print(f'{c["file"]}:{c["lo"]}-{c["hi"]} {d}  tok={toks}')
