#!/usr/bin/env bash
# check_sdpfull.sh — re-check the full-input LipSDP certificates, which no lib builds.
#
# `ScorecardSDPFull{,Uncon}` are CI-disabled: one PSD `linarith` goal needs ~15 GB, which kills
# the 16 GB runners (planning/archive/certs_heavy_psd_memory.md). So nothing re-checks them
# unless this is run. Run it after touching either file, `LipschitzCert/PairSDP`, the base
# scorecard files they import, or their generator (`lipschitz_cert_pair_sdp_full.py`).
#
# It builds the two modules ONE AT A TIME (~3 min and ~15 GB peak each, measured 2026-09-26),
# then prints `#print axioms` for every public theorem in both files and fails unless each rests
# on a subset of {propext, Classical.choice, Quot.sound}. Needs ~16 GB free; not for a laptop.
set -euo pipefail
cd "$(dirname "$0")/../.."
D=LeanMlir/Proofs/Certificates/LipschitzCert
for m in ScorecardSDPFull ScorecardSDPFullUncon; do
  echo "── lake build +LeanMlir.Proofs.Certificates.LipschitzCert.$m"
  lake build "+LeanMlir.Proofs.Certificates.LipschitzCert.$m"
done

tmp=$(mktemp --suffix=.lean)
trap 'rm -f "$tmp"' EXIT
{
  echo "import LeanMlir.Proofs.Certificates.LipschitzCert.ScorecardSDPFullUncon"
  for f in "$D/ScorecardSDPFull.lean" "$D/ScorecardSDPFullUncon.lean"; do
    python3 - "$f" <<'EOF'
import re, sys
ns = []
for line in open(sys.argv[1]):
    if m := re.match(r"^namespace (\S+)", line):
        ns.append(m.group(1))
    elif (m := re.match(r"^end (\S+)", line)) and ns and ns[-1] == m.group(1):
        ns.pop()
    elif m := re.match(r"^theorem (\S+)", line):
        print(f"#print axioms {'.'.join(ns + [m.group(1)])}")
EOF
  done
} > "$tmp"
n=$(grep -c '^#print' "$tmp")
out=$(lake env lean "$tmp" 2>&1 | sed ':a;N;$!ba;s/,\n /, /g')
bad=$(echo "$out" | grep -E 'depends on axioms|error|sorry' \
      | grep -vE "depends on axioms: \[(propext|Classical\.choice|Quot\.sound)(, (propext|Classical\.choice|Quot\.sound))*\]" || true)
if [ -n "$bad" ]; then
  echo "$bad" | head -20
  echo "✗ SDPFull axiom audit: the lines above are not on the standard axioms" >&2
  exit 1
fi
echo "✓ SDPFull: both modules build; $n theorems, each on a subset of propext / Classical.choice / Quot.sound"
