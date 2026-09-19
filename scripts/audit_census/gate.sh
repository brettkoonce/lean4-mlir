#!/usr/bin/env bash
# The full gate for a retirement batch (planning/audit_census.md):
#   Certs -> default build -> the LeanMlir lib root -> AuditAxioms (every #print line has a verdict,
#   all within {propext, Classical.choice, Quot.sound}) -> docstring-checkrefs -> both coverage
#   scripts -> verified_mlir/ byte-identical. Logs in $GATE_LOG. Ends with GATE: PASS or FAIL.
# ⚠ The default target is the Proofs lib, which does not include LeanMlir.lean: without the
#   LeanMlir-lib step, removing an import there leaves a stale LeanMlir.olean that checkrefs trips on.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
L=${GATE_LOG:-/tmp/audit_census/gate}
mkdir -p $L
fail=0
t0=$SECONDS
lake build Certs > $L/certs.log 2>&1; rc=$?; echo "certs: exit $rc ($(grep -c '^error' $L/certs.log) errors) $((SECONDS-t0))s $(tail -1 $L/certs.log)"; [ $rc = 0 ] || fail=1
t0=$SECONDS
lake build > $L/build.log 2>&1; rc=$?; echo "build: exit $rc $((SECONDS-t0))s $(tail -1 $L/build.log)"; [ $rc = 0 ] || fail=1
lake build LeanMlir > $L/leanmlir.log 2>&1; rc=$?; echo "LeanMlir lib: exit $rc $(tail -1 $L/leanmlir.log)"; [ $rc = 0 ] || fail=1
t0=$SECONDS
lake env lean tests/AuditAxioms.lean > $L/audit.log 2>&1; rc=$?
python3 - "$L/audit.log" <<'PY' || fail=1
import re,sys
t=open(sys.argv[1]).read()
errs=[l for l in t.splitlines() if 'error' in l]
t2=re.sub(r'\n\s+',' ',t)
v=re.findall(r"depends on axioms: \[([^\]]*)\]",t2)
nodeps=len(re.findall(r"does not depend on any axioms",t2))
ok={"propext","Classical.choice","Quot.sound"}
bad=[x for x in v if not {a.strip() for a in x.split(',')} <= ok]
pins=len(re.findall(r'^#print axioms',open('tests/AuditAxioms.lean').read(),re.M))
print(f"audit: {len(v)+nodeps} verdicts / {pins} #print lines, {len(bad)} bad, {len(errs)} error lines")
sys.exit(0 if (len(v)+nodeps==pins and not bad and not errs) else 1)
PY
echo "  (audit $((SECONDS-t0))s, exit $rc)"
lake exe docstring-checkrefs > $L/refs.log 2>&1; rc=$?; echo "checkrefs: exit $rc $(tail -1 $L/refs.log)"; [ $rc = 0 ] || fail=1
python3 scripts/check_audit_coverage.py > $L/cov.log 2>&1; rc=$?; echo "audit coverage: exit $rc"; [ $rc = 0 ] || fail=1
python3 scripts/check_render_coverage.py > $L/rcov.log 2>&1; rc=$?; echo "render coverage: exit $rc $(tail -1 $L/rcov.log)"; [ $rc = 0 ] || fail=1
st=$(git status --porcelain verified_mlir/ | wc -l); echo "verified_mlir changes: $st"; [ $st = 0 ] || fail=1
echo "GATE: $([ $fail = 0 ] && echo PASS || echo FAIL)"
