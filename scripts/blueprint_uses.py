#!/usr/bin/env python3
"""Keep the blueprint's `\\uses{...}` lines equal to the real Lean dependencies.

The dependency graph at /blueprint/dep_graph_document.html is drawn from the
`\\uses{...}` line of every theorem/definition block in blueprint/src/content.tex.
Those lines were hand-written for the original proof suite and drifted as the
proofs were refactored: audited 2026-09-21, 155 of 242 edges matched the Lean
dependencies, 87 named lemmas the proofs no longer touch and 100 real ones were
missing. This script makes the line a generated artifact instead.

    lake exe blueprint-checkdecls blueprint/lean_decls blueprint/lean_deps
    python3 scripts/blueprint_uses.py --check     # CI: exit 1 on any drift
    python3 scripts/blueprint_uses.py --fix       # rewrite the \\uses lines

`blueprint/lean_deps` (one `dep name` pair per line, Lean names) is written by
the second argument of `blueprint-checkdecls`: for every declaration the
blueprint cites it walks the constants the declaration's type and proof use,
expanding through this project's own helpers and stopping at other cited
declarations. A block's expected `\\uses` is the set of blocks those stop
points belong to; a block with no dependencies has no `\\uses` line.
"""
import argparse
import re
import sys
from collections import defaultdict

ENVS = ("theorem", "lemma", "definition", "axiom", "proposition", "corollary")
BLOCK = re.compile(
    r"\\begin\{(%s)\}(.*?)\\end\{\1\}" % "|".join(ENVS), re.S)
LABEL = re.compile(r"\\label\{([^}]*)\}")
LEAN = re.compile(r"\\lean\{([^}]*)\}")
LEANOK = re.compile(r"^[ \t]*\\leanok[ \t]*\n", re.M)
USES = re.compile(r"^[ \t]*\\uses\{([^}]*)\}[ \t]*\n", re.M | re.S)
WIDTH = 88


def parse_blocks(tex):
    """-> [(start, end, label, [lean names], uses-set)] in document order."""
    out = []
    for m in BLOCK.finditer(tex):
        body = m.group(2)
        lab = LABEL.search(body)
        ln = LEAN.search(body)
        if not lab or not ln:
            continue
        names = [n.strip() for n in ln.group(1).split(",") if n.strip()]
        u = USES.search(body)
        uses = set(x.strip() for x in u.group(1).split(",") if x.strip()) if u else set()
        out.append((m.start(), m.end(), lab.group(1), names, uses))
    return out


def expected_uses(blocks, deps_path):
    lean2lab = {n: lab for _, _, lab, names, _ in blocks for n in names}
    labels = set(lab for _, _, lab, _, _ in blocks)
    exp = defaultdict(set)
    with open(deps_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) != 2:
                continue
            dep, name = parts
            if dep in lean2lab and name in lean2lab and lean2lab[dep] != lean2lab[name]:
                exp[lean2lab[name]].add(lean2lab[dep])
    return exp, labels


def render_uses(indent, labs):
    """One `\\uses{...}` line, wrapped at WIDTH with a 4-space continuation."""
    items = sorted(labs)
    lines, cur = [], indent + "\\uses{" + items[0]
    for it in items[1:]:
        if len(cur) + 2 + len(it) + 1 > WIDTH:
            lines.append(cur + ",")
            cur = indent + "    " + it
        else:
            cur += ", " + it
    lines.append(cur + "}")
    return "\n".join(lines) + "\n"


def fix_block(body, want):
    """Return the block body with its \\uses line set to `want` (a set of labels)."""
    have = USES.search(body)
    if not want:
        return USES.sub("", body, count=1) if have else body
    lean_line = re.search(r"^([ \t]*)\\lean\{", body, re.M)
    indent = lean_line.group(1) if lean_line else "  "
    new = render_uses(indent, want)
    if have:
        return body[:have.start()] + new + body[have.end():]
    anchor = LEANOK.search(body) or re.search(r"^[ \t]*\\lean\{[^}]*\}[ \t]*\n", body, re.M)
    return body[:anchor.end()] + new + body[anchor.end():]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tex", default="blueprint/src/content.tex")
    ap.add_argument("--deps", default="blueprint/lean_deps")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="report drift; exit 1 if any")
    mode.add_argument("--fix", action="store_true", help="rewrite the \\uses lines in place")
    a = ap.parse_args()

    tex = open(a.tex).read()
    blocks = parse_blocks(tex)
    exp, labels = expected_uses(blocks, a.deps)

    drift = []
    for _, _, lab, _, uses in blocks:
        unknown = uses - labels
        if unknown:
            print(f"warning: {lab} uses unknown label(s): {', '.join(sorted(unknown))}")
        if uses != exp[lab]:
            drift.append((lab, uses - exp[lab], exp[lab] - uses))

    n_edges = sum(len(v) for v in exp.values())
    if a.check:
        for lab, spurious, missing in drift:
            print(f"{lab}:" + "".join(f"  -{x}" for x in sorted(spurious))
                  + "".join(f"  +{x}" for x in sorted(missing)))
        if drift:
            print(f"\n{len(drift)} of {len(blocks)} blocks drifted from the Lean dependencies "
                  f"(- listed but unused, + used but unlisted); run scripts/blueprint_uses.py --fix")
            return 1
        print(f"blueprint \\uses in sync with Lean: {len(blocks)} blocks, {n_edges} edges")
        return 0

    # --fix: rewrite blocks back to front so earlier offsets stay valid
    out = tex
    for start, end, lab, _, uses in reversed(blocks):
        if uses == exp[lab]:
            continue
        m = BLOCK.match(tex, start)
        body = m.group(2)
        new_body = fix_block(body, exp[lab])
        out = out[:start + len(m.group(0)) - len(body) - len(f"\\end{{{m.group(1)}}}")] + new_body \
            + out[start + len(m.group(0)) - len(f"\\end{{{m.group(1)}}}"):]
    open(a.tex, "w").write(out)
    print(f"rewrote {len(drift)} of {len(blocks)} blocks in {a.tex} ({n_edges} edges)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
