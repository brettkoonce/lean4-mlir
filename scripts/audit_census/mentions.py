"""Every remaining mention (code, docstrings, docs) of the short names a retirement group cuts.

    CENSUS_DIR=... python3 scripts/audit_census/mentions.py groups.json <census dir>

Run after retire.py --apply; a hit is prose to re-point or a leftover to cut."""
import sys, re, subprocess, json, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["CENSUS_DIR"] = sys.argv[2]
import size as S
g = json.load(open(sys.argv[1]))
seed, follow, keep = [], [], []
for v in g.values(): seed += v["seed"]; follow += v.get("follow", []); keep += v.get("keep", [])
r = S.size(seed, follow, keep)
cut = r["pins_cut"] + r["unpinned_cut"]
def short(n):
    c = n.split(".")
    if n.startswith("_private."): c = c[c.index("0")+1:]
    return c[-1]
names = sorted({short(n) for n in cut})
# names still declared elsewhere (homonyms) are ambiguous; flag
files = subprocess.run(["git", "ls-files"], capture_output=True, text=True).stdout.split()
files = [f for f in files if not f.startswith(("runs/", "planning/archive/")) and re.search(r"\.(lean|md|tex|ya?ml|py|sh|txt)$", f)]
hits = {}
for f in files:
    try: t = open(f, encoding="utf-8", errors="replace").read()
    except Exception: continue
    t2 = t.replace("\\_", "_")
    for n in names:
        for m in re.finditer(r"(?<![\w'])" + re.escape(n) + r"(?![\w'!?])", t2):
            ln = t2.count("\n", 0, m.start()) + 1
            hits.setdefault((f, ln), set()).add(n)
for (f, ln), ns in sorted(hits.items()):
    line = open(f, encoding="utf-8", errors="replace").read().split("\n")[ln-1].strip()
    print(f"{f}:{ln}: [{', '.join(sorted(ns))}] {line[:130]}")
