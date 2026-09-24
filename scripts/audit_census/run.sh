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
import os, sys
from pathlib import Path
sys.path.insert(0, "scripts")
from lean_graph import lakefile_text, lib_roots, libs, reachable
out = Path(os.environ["CENSUS_DIR"])
text = lakefile_text()
roots, gate_roots = [], []
for name in libs(text):
    if "roots" not in text.split(f"lean_lib {name} where", 1)[1].split("lean_lib", 1)[0]: continue
    rs = [r for r in lib_roots(text, name) if not r.startswith(("apps", "demos"))]
    roots += rs
    if "CertsHeavy" not in name: gate_roots += rs
(out / "roots.txt").write_text("\n".join(dict.fromkeys(gate_roots)) + "\n")
# every LeanMlir module reachable from a lib root that has an olean
seen = reachable(roots, strict=False)
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
