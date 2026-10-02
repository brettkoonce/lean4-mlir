#!/usr/bin/env python3
"""The shape of the repo: every tracked file matches a line of scripts/gates/repo_shape.txt.

A file no line allows fails, and so does a file over its size cap (2 MB unless its line
says otherwise). Old files that predate a rule are listed by `grandfather` lines, which can
only shrink: one that no longer matches a tracked file fails too, so it gets deleted with
the file. An exact (glob-free) allow line that matches nothing fails the same way, which
keeps the root listing exact.

It reads the INDEX (git ls-files, blob sizes from git cat-file), so the pre-commit hook in
.githooks/ checks what is about to be committed, not the working tree.

    python3 scripts/gates/repo_shape.py              # check
    python3 scripts/gates/repo_shape.py --unmatched  # print what fails, as grandfather lines

Changing repo_shape.txt is the user's call: an agent asks before adding a line.
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SHAPE = ROOT / "scripts/gates/repo_shape.txt"
DEFAULT_CAP = 2_000_000
UNITS = {"K": 1_000, "M": 1_000_000}


def glob_re(pat: str) -> re.Pattern:
    """`**/` any directories (or none), `**` anything, `*` one segment's characters, `?` one
    character, `{a,b}` alternatives."""
    out, i = "", 0
    while i < len(pat):
        if pat.startswith("**/", i):
            out, i = out + "(?:.*/)?", i + 3
        elif pat.startswith("**", i):
            out, i = out + ".*", i + 2
        elif pat[i] == "*":
            out, i = out + "[^/]*", i + 1
        elif pat[i] == "?":
            out, i = out + "[^/]", i + 1
        elif pat[i] == "{":
            j = pat.index("}", i)
            out += "(?:" + "|".join(re.escape(a) for a in pat[i + 1:j].split(",")) + ")"
            i = j + 1
        else:
            out, i = out + re.escape(pat[i]), i + 1
    return re.compile(out + r"\Z")


def expand(pat: str) -> list[str]:
    m = re.search(r"\{([^}]*)\}", pat)
    if not m:
        return [pat]
    return [x for a in m[1].split(",") for x in expand(pat[:m.start()] + a + pat[m.end():])]


def load():
    allow, grand = [], []
    for n, raw in enumerate(SHAPE.read_text().splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        words = line.split()
        if words[0] == "grandfather":
            grand.append((n, words[1], glob_re(words[1])))
            continue
        cap = DEFAULT_CAP
        for w in words[1:]:
            m = re.fullmatch(r"cap=(\d+)([KM])", w)
            if not m:
                sys.exit(f"repo_shape.txt:{n}: cannot read {w!r}")
            cap = int(m[1]) * UNITS[m[2]]
        allow.append((n, words[0], glob_re(words[0]), cap))
    return allow, grand


def tracked() -> dict[str, int]:
    ls = subprocess.run(["git", "ls-files", "-s", "-z"], cwd=ROOT, capture_output=True,
                        check=True).stdout.decode().split("\0")
    entries = [(e.split()[1], e.split("\t", 1)[1]) for e in ls if e]
    batch = subprocess.run(["git", "cat-file", "--batch-check=%(objectsize)"], cwd=ROOT,
                           input="\n".join(o for o, _ in entries).encode(),
                           capture_output=True, check=True).stdout.decode().split()
    return {path: int(size) for (_, path), size in zip(entries, batch)}


def main() -> int:
    allow, grand = load()
    files = tracked()
    bad, gused = [], set()
    for path, size in sorted(files.items()):
        g = next((n for n, _, rx, in grand if rx.match(path)), None)
        if g is not None:
            gused.add(g)
            continue
        hit = next(((n, cap) for n, _, rx, cap in allow if rx.match(path)), None)
        if hit is None:
            bad.append((path, "no line allows it"))
            continue
        if size > hit[1]:
            bad.append((path, f"{size / 1e6:.1f} MB, over the line's {hit[1] / 1e6:g} MB cap"))
    stale = [(n, p, "grandfather line matches no tracked file") for n, p, _ in grand if n not in gused]
    for n, p, _, _ in allow:
        if not re.search(r"[*?]", p):  # exact: each brace alternative must name a tracked file
            stale += [(n, alt, "exact allow line matches no tracked file")
                      for alt in expand(p) if alt not in files]
    if "--unmatched" in sys.argv:
        for path, _ in bad:
            print(f"grandfather {path}")
        return 0
    for path, why in bad:
        print(f"✗ {path}: {why}")
    for n, p, why in stale:
        print(f"✗ repo_shape.txt:{n} {p}: {why}")
    if bad or stale:
        print(f"\n{len(bad) + len(stale)} problem(s). A new kind of file needs a line in "
              "scripts/gates/repo_shape.txt, which is the user's call; a run goes in "
              "runs/YYYY-MM-DD-<slug>/.")
        return 1
    print(f"✓ {len(files)} tracked files match scripts/gates/repo_shape.txt "
          f"({sum(1 for p in files if any(rx.match(p) for _, _, rx in grand))} grandfathered)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
