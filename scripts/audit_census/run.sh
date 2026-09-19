#!/usr/bin/env bash
# Audit-only census + docstring citation survey (planning/audit_census.md).
# Needs a current `lake build Certs` (and CertsHeavy/Codegen/ProofsMinimal for the full module set).
#   scripts/audit_census/run.sh            # use graph + audit-only list      (~2.5 min)
#   scripts/audit_census/run.sh refs       # also every docstring citation    (+1.5 min)
# Output in $CENSUS_DIR (default /tmp/audit_census). Size a retirement group with
#   CENSUS_DIR=... python3 scripts/audit_census/size.py groups.json sized.json
# where groups.json is {"name": {"seed": [pins...], "follow": [orphaned pins to take too]}}.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
export CENSUS_DIR="${CENSUS_DIR:-/tmp/audit_census}"
mkdir -p "$CENSUS_DIR"
python3 - <<'PY'
import os, re
from pathlib import Path
out = Path(os.environ["CENSUS_DIR"])
text = Path("lakefile.lean").read_text()
roots, gate_roots = [], []
for seg in re.split(r"\nlean_lib ", text)[1:]:
    name = seg.split()[0]
    seg = seg.split("lean_exe", 1)[0].split("\nlean_lib", 1)[0]
    body = seg.split("roots", 1)
    if len(body) < 2: continue
    code = "\n".join(l.split("--", 1)[0] for l in body[1].split("]", 1)[0].splitlines())
    rs = [r for r in re.findall(r"`([A-Za-z0-9_.«»]+)", code) if not r.startswith(("apps", "demos"))]
    roots += rs
    if "CertsHeavy" not in name: gate_roots += rs
(out / "roots.txt").write_text("\n".join(dict.fromkeys(gate_roots)) + "\n")
# every LeanMlir module reachable from a lib root that has an olean
seen, st = set(), [r for r in roots if r.startswith("LeanMlir")]
while st:
    m = st.pop()
    p = Path(m.replace(".", "/") + ".lean")
    if m in seen or not p.exists(): continue
    seen.add(m)
    st += [x for x in re.findall(r"^import\s+([A-Za-z0-9_.]+)", p.read_text(), re.M) if x.startswith("LeanMlir")]
mods = sorted(m for m in seen if Path(".lake/build/lib/lean/" + m.replace(".", "/") + ".olean").exists())
(out / "modules.txt").write_text("\n".join(mods) + "\n")
print(f"{len(mods)} modules, {len(gate_roots)} gate roots")
PY
lake env lean --run scripts/audit_census/Dump.lean "$CENSUS_DIR/modules.txt" "$CENSUS_DIR/decls.tsv"
python3 scripts/audit_census/graph.py
python3 scripts/audit_census/census.py | head -3
if [[ "${1:-}" == refs ]]; then
  ALLREFS_ROOTS="$CENSUS_DIR/roots.txt" ALLREFS_OUT="$CENSUS_DIR/allmisses.tsv" \
    lake env lean --run scripts/audit_census/AllRefs.lean | head -2
fi
