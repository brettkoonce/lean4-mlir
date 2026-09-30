#!/usr/bin/env python3
"""Report the declarations whose only consumer is the axiom audit.

`tests/AuditAxioms.lean` (and `AuditAxiomsHeavy.lean`) `#print axioms` a declaration, which keeps
it compiled and axiom-clean but is not a use: a theorem nothing cites, that is absent from the
book, `formalization.yaml`, the comparator and the READMEs, can sit there indefinitely. This
lists every audited declaration whose only mention outside its defining file is an AuditAxioms
file, so a human can decide whether it is roadmap (cite it somewhere) or dead (delete it and its
pin).

A report, not a gate: it prints the list and exits 0.

Two sets. The literal one (`--literal`): no mention outside the defining file but the audit's.
It is large, because most lemmas are consumed inside their own file. The default is the
reachability one, which is how A-scope-1's cluster hid: a declaration is USED when a root
cites it (a non-Lean file, the comparator's challenge files, a Lean module header, a
`#guard`/`#eval`/`example` command) or when
the body of a used declaration does; the report lists audited declarations nothing uses.

What counts as a mention: the declaration's last name component as a whole identifier in any
tracked `.lean`, `.tex`, `.yaml`/`.yml`, `.py`, `.sh` or `.md` file, except the defining file,
the two AuditAxioms files, and the planning/, historical/ and runs/ trees (plans and records name
things without consuming them). Matching on the last component over-counts when two
declarations share it, so a colliding name can hide from the report; it never lists a
declaration that has a real consumer.

    python3 scripts/gates/audit_only_mentions.py            # the list
    python3 scripts/gates/audit_only_mentions.py --where    # plus each one's defining file
    python3 scripts/gates/audit_only_mentions.py --literal  # the literal set instead
"""
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUDITS = [ROOT / "tests" / "AuditAxioms.lean", ROOT / "tests" / "AuditAxiomsHeavy.lean"]
EXTS = (".lean", ".tex", ".yaml", ".yml", ".py", ".sh", ".md")
SKIP_DIRS = ("planning/", "historical/", "runs/")
COMPARATOR = "tests/comparator/"   # restates statements to check them: every mention is a use

DECL = re.compile(
    r"^\s*(?:@\[[^\]]*\]\s*)*(?:(?:private|protected|noncomputable|partial|unsafe|nonrec)\s+)*"
    r"(?:theorem|lemma|def|abbrev|instance|structure|inductive|class|opaque|axiom)\s+"
    r"([^\s:({\[]+)", re.M)
IDENT = re.compile(r"[\w'!?.]+")
PRINT = re.compile(r"^#print\s+axioms\s+(\S+)", re.M)


def tracked() -> list[str]:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True,
                         check=True).stdout.split("\n")
    return [p for p in out if p.endswith(EXTS) and not p.startswith(SKIP_DIRS)]


def last(name: str) -> str:
    return name.rstrip(".").split(".")[-1].lstrip("«").rstrip("»")


ROOTCMD = re.compile(r"^(?:#guard|#eval|#check|#print|#reduce|example\b|run_cmd|run_meta)", re.M)


def idents(text: str) -> set[str]:
    return {c for tok in IDENT.findall(text) for c in tok.split(".") if c}


def segments(text: str):
    """Split a Lean file at each declaration and each top-level command. Yields
    `(declared last component or None, text)`; `None` is a root segment (module header,
    `#guard`/`#eval`/`example` commands), whose mentions are uses."""
    cuts = [(m.start(), last(m.group(1))) for m in DECL.finditer(text)]
    cuts += [(m.start(), None) for m in ROOTCMD.finditer(text)]
    cuts.sort()
    prev, name = 0, None
    for pos, nxt in cuts:
        yield name, text[prev:pos]
        prev, name = pos, nxt
    yield name, text[prev:]


def main() -> None:
    files = tracked()
    audit_rel = {str(a.relative_to(ROOT)) for a in AUDITS}
    defined: dict[str, set[str]] = defaultdict(set)     # last component -> defining files
    mentions: dict[str, set[str]] = defaultdict(set)    # last component -> files mentioning it
    body: dict[str, set[str]] = defaultdict(set)        # last component -> names its bodies cite
    roots: set[str] = set()                             # names a root segment cites
    for rel in files:
        if rel in audit_rel:
            continue
        try:
            text = (ROOT / rel).read_text(errors="replace")
        except (FileNotFoundError, IsADirectoryError):
            continue
        for comp in idents(text):
            mentions[comp].add(rel)
        if not rel.endswith(".lean") or rel.startswith(COMPARATOR):
            roots |= idents(text)
            continue
        for name, seg in segments(text):
            ids = idents(seg)
            if name is None:
                roots |= ids
            else:
                defined[name].add(rel)
                body[name] |= ids - {name}

    # Live = cited by a root, or by the body of a live declaration (names resolved by last
    # component, so a collision can only make something look live).
    live, todo = set(), [n for n in roots if n in defined]
    while todo:
        n = todo.pop()
        if n in live:
            continue
        live.add(n)
        todo += [m for m in body.get(n, ()) if m in defined and m not in live]

    audited = []
    for a in AUDITS:
        if a.exists():
            audited += PRINT.findall(a.read_text())
    audited = list(dict.fromkeys(audited))
    literal, dead = [], []
    for name in audited:
        short = last(name)
        homes = defined.get(short, set())
        if not homes:
            continue
        if not (mentions.get(short, set()) - homes):
            literal.append((name, sorted(homes)))
        if short not in live:
            dead.append((name, sorted(homes)))

    where = "--where" in sys.argv
    rows = literal if "--literal" in sys.argv else dead
    print(f"{len(literal)} of {len(audited)} audited declarations are mentioned outside their "
          f"defining file only by AuditAxioms; {len(dead)} are reached by no use at all (no "
          f"root cites them or a declaration that is reached). Listing the "
          f"{'first' if '--literal' in sys.argv else 'second'} set:")
    for name, homes in sorted(rows, key=lambda t: (t[1], t[0])):
        print(f"  {name}" + (f"    ({', '.join(homes)})" if where else ""))


if __name__ == "__main__":
    main()
