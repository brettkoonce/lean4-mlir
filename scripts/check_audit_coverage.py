#!/usr/bin/env python3
"""Assert every tests/AuditAxioms.lean import is reachable from the Proofs roots.

CI's axiom gate runs `lake env lean tests/AuditAxioms.lean` against whatever
`lake build Proofs` produced. If an audited module is not a Proofs root (or a
transitive import of one), its .olean never gets built on a fresh runner and
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
    """Extract the `Proofs` lib's roots, ignoring `--` comments (which contain
    brackets like [3,4,6,3] that defeat naive `roots := #[...]` matching)."""
    try:
        segment = text.split("lean_lib «Proofs» where", 1)[1]
    except IndexError:
        sys.exit("error: no `lean_lib «Proofs»` in lakefile.lean")
    segment = segment.split("lean_lib", 1)[0]
    code = "\n".join(line.split("--", 1)[0] for line in segment.splitlines())
    return re.findall(r"`([A-Za-z0-9_.]+)", code)


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


def main() -> None:
    covered = reachable(proofs_roots(LAKEFILE.read_text()))
    audited = re.findall(r"^import\s+([A-Za-z0-9_.]+)", AUDIT.read_text(), re.M)
    missing = [m for m in audited if m.startswith("LeanMlir") and m not in covered]
    if missing:
        print(f"error: {AUDIT} imports module(s) not reachable from the "
              f"`Proofs` lib roots in {LAKEFILE}:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        print("\nfix: add the apex module(s) to `lean_lib «Proofs»`'s roots "
              "(an apex that transitively imports the rest suffices).",
              file=sys.stderr)
        sys.exit(1)
    print(f"audit coverage OK: all {len(audited)} AuditAxioms imports reachable "
          f"from the Proofs roots ({len(covered)} modules covered)")


if __name__ == "__main__":
    main()
