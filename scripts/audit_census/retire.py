"""Retire a group of declarations by their elaborated declaration ranges.

    python3 scripts/audit_census/retire.py groups.json            # dry run: the cut set + citations
    python3 scripts/audit_census/retire.py groups.json --apply    # cut, and drop their #print axioms lines

groups.json is size.py's input ({"name": {"seed": [...], "follow": [...], "keep": [...]}}); the cut set is size.py's
fixed point over all groups at once, so ranges come from the current $CENSUS_DIR graph (re-run
run.sh after every cut). A declaration's range starts at its docstring; a `set_option ... in` line
directly above it goes too. Refuses to apply when a cut name is cited by the book, formalization.yaml,
certs.yml or a README (re-point those first).
"""
import json, re, sys
from collections import defaultdict
from pathlib import Path
import size as S  # loads the $CENSUS_DIR graph
import subprocess

REPO = Path(subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True,
                           text=True).stdout.strip())

def short(n):
    comps = n.split(".")
    if n.startswith("_private."):
        comps = comps[comps.index("0") + 1:]
    return comps[-1]

def cited_outside(names):
    """short name -> [files] among the book / yaml / certs.yml / READMEs."""
    files = ["blueprint/src/content.tex", "formalization.yaml", ".github/workflows/certs.yml"]
    files += subprocess.run(["git", "ls-files", "*README.md"], cwd=REPO, capture_output=True,
                            text=True).stdout.split()
    texts = {f: (REPO / f).read_text(encoding="utf-8", errors="replace").replace("\\_", "_")
             for f in files if (REPO / f).exists()}
    out = defaultdict(list)
    for n in names:
        s = short(n)
        user = n.split(".0.", 1)[1] if n.startswith("_private.") else n
        # a qualified citation (`Back.subst`) only counts when its qualifier is this name's tail
        rx = re.compile(r"(?<![\w'.])((?:[\w']+\.)*)" + re.escape(s) + r"(?![\w'!?])")
        for f, t in texts.items():
            if any(not q or user.endswith("." + q + s) or user == q + s
                   for q in (m.group(1) for m in rx.finditer(t))):
                out[n].append(f)
    return out

def main():
    groups = json.load(open(sys.argv[1]))
    apply = "--apply" in sys.argv
    seed, follow, keep = [], [], []
    for g in groups.values():
        seed += g["seed"]; follow += g.get("follow", []); keep += g.get("keep", [])
    r = S.size(seed, follow, keep)
    # never cut a declaration nested inside another one's range (a structure field, a
    # constructor): structure-instance syntax sets a field without naming its projection, so the
    # graph can call a used field unused
    ranges = defaultdict(list)
    for n, c in S.consts.items():
        if c["lo"] is not None:
            ranges[c["file"]].append((c["lo"], c["hi"], n))
    nested = set()
    for n in r["pins_cut"] + r["unpinned_cut"]:
        c = S.consts[n]
        if any(lo <= c["lo"] and c["hi"] <= hi and (lo, hi) != (c["lo"], c["hi"])
               for lo, hi, m in ranges[c["file"]]):
            nested.add(n)
    if nested:
        print("skipped (nested in another declaration):", sorted(nested))
        keep = keep + sorted(nested)
        r = S.size(seed, follow, keep)
    cut = r["pins_cut"] + r["unpinned_cut"]
    print(f"{len(cut)} declarations, {r['lines']} lines, {r['pins']} pins, {len(r['files'])} files")
    if r["newly_audit_only_pins"]:
        print("pins left without a user (kept):", r["newly_audit_only_pins"])
    by_file = defaultdict(list)
    for n in cut:
        c = S.consts[n]
        by_file[c["file"]].append((c["lo"], c["hi"], n))
    for f in sorted(by_file):
        print(f"  {f}")
        for lo, hi, n in sorted(by_file[f]):
            print(f"    {lo:5d}-{hi:<5d} {'PIN ' if n in S.pinset else '    '}{n}")
    cites = cited_outside(cut)
    if cites:
        print("CITED outside Lean:")
        for n, fs in cites.items():
            print(f"  {n}: {fs}")
    if not apply:
        return
    if cites and "--force-cited" not in sys.argv:
        sys.exit("refusing: re-point the citations above first (or pass --force-cited)")
    for f, rs in by_file.items():
        p = REPO / f
        lines = p.read_text(encoding="utf-8").split("\n")
        drop = set()
        for lo, hi, _ in rs:
            a = lo
            while a - 2 >= 0 and re.match(r"^\s*set_option\s.*\sin\s*$", lines[a - 2]):
                a -= 1
            drop.update(range(a, hi + 1))            # 1-based
        # a cut between two blank lines leaves one blank line, not two
        blank = lambda i: 1 <= i <= len(lines) and not lines[i - 1].strip()
        for e in sorted(i for i in drop if i + 1 not in drop):
            s = e
            while s - 1 in drop: s -= 1
            if (s == 1 or blank(s - 1)) and blank(e + 1):
                drop.add(e + 1)
        kept_lines = [l for i, l in enumerate(lines, 1) if i not in drop]
        p.write_text("\n".join(kept_lines), encoding="utf-8")
    pins = set(r["pins_cut"])
    ap = REPO / "tests/AuditAxioms.lean"
    out, n_rm = [], 0
    for l in ap.read_text(encoding="utf-8").split("\n"):
        m = re.match(r"#print axioms\s+(\S+)\s*$", l)
        if m and (m.group(1) in pins or "Proofs." + m.group(1) in pins):
            n_rm += 1
            continue
        out.append(l)
    ap.write_text("\n".join(out), encoding="utf-8")
    print(f"applied: {len(by_file)} files; {n_rm} #print axioms lines removed")

if __name__ == "__main__":
    main()
