#!/usr/bin/env python3
"""Assert every tests/AuditAxioms.lean import is reachable from the proof-lib roots.

CI's axiom gate (certs.yml) runs `lake env lean tests/AuditAxioms.lean` against
whatever `lake build Certs` produced (whose roots subsume the per-push `Proofs`
slice). If an audited module is not a root of either lib (or a transitive
import of one), its .olean never gets built on a fresh runner and
the gate dies on a missing object file — silently, because the workflow step's
`set -e` kills the script at the capture assignment. Locally the gap hides
behind stale .oleans from dev builds, so this is exactly the failure that only
shows up in CI (bit us at 5f27766^).

This check is pure text analysis (no Lean, no build): parse the roots out of
lakefile.lean, BFS the `import` graph over the .lean sources, and demand the
audit's imports are covered. Run it from the repo root; exits 1 with the gap
and the fix when coverage is missing.
"""

import re
import sys
from pathlib import Path

LAKEFILE = Path("lakefile.lean")
AUDIT = Path("tests/AuditAxioms.lean")


def proofs_roots(text: str) -> list[str]:
    """Extract the union of the `Proofs` (per-push IR/render slice) and `Certs`
    (certificate corpus) libs' roots, ignoring `--` comments (which contain
    brackets like [3,4,6,3] that defeat naive `roots := #[...]` matching)."""
    roots: list[str] = []
    for lib in ("«Proofs»", "«Certs»"):
        try:
            segment = text.split(f"lean_lib {lib} where", 1)[1]
        except IndexError:
            sys.exit(f"error: no `lean_lib {lib}` in lakefile.lean")
        segment = segment.split("lean_lib", 1)[0]
        code = "\n".join(line.split("--", 1)[0] for line in segment.splitlines())
        roots += re.findall(r"`([A-Za-z0-9_.]+)", code)
    return roots


def imports_of(module: str) -> list[str]:
    path = Path(module.replace(".", "/") + ".lean")
    if not path.exists():
        sys.exit(f"error: root/import `{module}` has no source file at {path}")
    return re.findall(r"^import\s+([A-Za-z0-9_.]+)", path.read_text(), re.M)


def reachable(roots: list[str]) -> set[str]:
    seen: set[str] = set()
    stack = [r for r in roots if r.startswith("LeanMlir")]
    while stack:
        module = stack.pop()
        if module in seen:
            continue
        seen.add(module)
        stack += [m for m in imports_of(module) if m.startswith("LeanMlir")]
    return seen


def lib_roots(text: str, lib: str) -> list[str]:
    """Roots of one lib (comment-stripped), e.g. «CertsHeavy»."""
    segment = text.split(f"lean_lib {lib} where", 1)[1].split("lean_lib", 1)[0]
    code = "\n".join(line.split("--", 1)[0] for line in segment.splitlines())
    return re.findall(r"`([A-Za-z0-9_.]+)", code)


def check(audit: Path, covered: set[str], libs_desc: str, fix: str) -> int:
    audited = re.findall(r"^import\s+([A-Za-z0-9_.]+)", audit.read_text(), re.M)
    missing = [m for m in audited if m.startswith("LeanMlir") and m not in covered]
    if missing:
        print(f"error: {audit} imports module(s) not reachable from the "
              f"{libs_desc} lib roots in {LAKEFILE}:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        print(f"\nfix: {fix}", file=sys.stderr)
        sys.exit(1)
    return len(audited)


def main() -> None:
    text = LAKEFILE.read_text()
    covered = reachable(proofs_roots(text))
    n = check(AUDIT, covered, "`Proofs`/`Certs`",
              "add the apex module(s) to `lean_lib «Certs»`'s roots "
              "(an apex that transitively imports the rest suffices) — or, for "
              "data-heavy generated instances, to `CertsHeavy` + AuditAxiomsHeavy.")
    print(f"audit coverage OK: all {n} AuditAxioms imports reachable "
          f"from the Proofs+Certs roots ({len(covered)} modules covered)")
    heavy_audit = Path("tests/AuditAxiomsHeavy.lean")
    if heavy_audit.exists():
        covered_heavy = reachable(proofs_roots(text) + lib_roots(text, "«CertsHeavy»"))
        nh = check(heavy_audit, covered_heavy, "`Proofs`/`Certs`/`CertsHeavy`",
                   "add the apex module(s) to `lean_lib «CertsHeavy»`'s roots.")
        print(f"heavy audit coverage OK: all {nh} AuditAxiomsHeavy imports reachable "
              f"from the Proofs+Certs+CertsHeavy roots ({len(covered_heavy)} modules covered)")


if __name__ == "__main__":
    main()
