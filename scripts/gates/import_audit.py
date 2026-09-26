#!/usr/bin/env python3
"""Import audit — implied and unused imports (planning/placement_imports_cleanup.md §9).

    python3 scripts/gates/import_audit.py implied            # CI: exit 1 on any implied import
    python3 scripts/gates/import_audit.py implied --all      # also report tests/
    python3 scripts/gates/import_audit.py unused             # candidates, from the built .oleans
    python3 scripts/gates/import_audit.py unused --verify    # ... each confirmed by a compile

`implied`: an import that another import of the same file already reaches. Removing one leaves
every closure unchanged, so it is exact and needs no Lean: the import graph is read from the
sources, Mathlib's included (from `.lake/packages`). Checked for `LeanMlir/`, `apps/` and
`demos/`, which are at zero; `tests/` lists some imports on purpose (`AuditAxioms`, the comparator
tier) and is reported only with `--all`.

`unused`: an import none of whose closure defines a constant the file's constants use, beyond
what its other imports reach. `ImportAudit.lean` reads the constants from the `.olean`s, so run
it after `lake build`. Only modules some lean_lib builds are audited (not the SDPFull pair).
Constants are not the whole story — a tactic, a notation, an `#eval` or an `example` leaves no
constant behind — so a candidate is only an import the file compiles without. `--verify`
compiles a scratch copy of each candidate's file with that one import removed; a file with a
top-level `#eval` is not compiled (it would run). A verified removal can still break a DOWNSTREAM
file that reached something through it: rebuild the libs after acting on the list.
"""
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.setrecursionlimit(100000)

IMP = re.compile(r'^(?:public\s+|meta\s+|private\s+)*import\s+(?:all\s+)?([\w.«»]+)')
PROJECT_DIRS = ["LeanMlir", "apps", "demos"]
CHECKED_PREFIXES = ("LeanMlir/", "LeanMlir.lean", "apps/", "demos/")


def parse_imports(path):
    out = []
    with open(path, errors="ignore") as fh:
        for line in fh:
            s = line.strip()
            m = IMP.match(s)
            if m:
                out.append(m.group(1))
                continue
            if s and not s.startswith(("--", "/-", "module", "prelude")) and out:
                break
    return tuple(out)


def project_files(with_tests):
    files = ["LeanMlir.lean"]
    for d in PROJECT_DIRS + (["tests"] if with_tests else []):
        for dp, _, fs in os.walk(d):
            if ".lake" in dp:
                continue
            files += [os.path.join(dp, f) for f in fs if f.endswith(".lean")]
    return sorted(files)


def module_of(path):
    return path[:-5].replace("/", ".")


@lru_cache(None)
def source_index():
    """module -> source path, for the project and every package under .lake/packages."""
    src = {}
    pk = ".lake/packages"
    if os.path.isdir(pk):
        for pkg in os.listdir(pk):
            b = os.path.join(pk, pkg)
            for d in os.listdir(b):
                if d[0].isupper() and os.path.isdir(os.path.join(b, d)):
                    for dp, _, fs in os.walk(os.path.join(b, d)):
                        for f in fs:
                            if f.endswith(".lean"):
                                p = os.path.join(dp, f)
                                src[os.path.relpath(p, b)[:-5].replace("/", ".")] = p
    for f in project_files(True):
        src[module_of(f)] = f
    return src


@lru_cache(None)
def imports_of(mod):
    p = source_index().get(mod)
    return parse_imports(p) if p else ()


@lru_cache(None)
def closure(mod):
    c = set()
    for i in imports_of(mod):
        c.add(i)
        c |= closure(i)
    return frozenset(c)


def implied(with_tests):
    if not os.path.isdir(".lake/packages/mathlib"):
        print("note: .lake/packages/mathlib is absent; Mathlib-to-Mathlib edges are not seen",
              file=sys.stderr)
    bad = []
    for f in project_files(with_tests):
        if f == "LeanMlir.lean":
            continue                     # the doc-gen umbrella lists modules on purpose
        imps = imports_of(module_of(f))
        red = [x for x in imps if any(x in closure(y) for y in imps if y != x)]
        if red:
            bad.append((f, red))
    checked = [(f, r) for f, r in bad if f.startswith(CHECKED_PREFIXES)]
    for f, red in bad:
        tag = "" if f.startswith(CHECKED_PREFIXES) else " (not gated)"
        print(f"{f}{tag}: {' '.join(red)}")
    n = sum(len(r) for _, r in checked)
    if n:
        print(f"\nerror: {n} implied import(s) — each is reached through another import of the "
              "same file; delete it (closures are unchanged).", file=sys.stderr)
        return 1
    print(f"✓ no implied imports in {', '.join(PROJECT_DIRS)} "
          f"({len(project_files(False))} files)")
    return 0


def olean(mod):
    return os.path.join(".lake/build/lib/lean", mod.replace(".", "/") + ".olean")


def required_modules(mods, stale):
    """Run ImportAudit.lean; an `.olean` it cannot read (an old toolchain's, for a module no lib
    builds any more) is dropped into `stale` and the run retried without it."""
    mods = list(mods)
    while True:
        out = subprocess.run(["lake", "env", "lean", "--run", "scripts/gates/ImportAudit.lean", *mods],
                             capture_output=True, text=True)
        if out.returncode == 0:
            break
        m = re.search(r"failed to read file '[^']*/lib/lean/([^']+)\.olean'", out.stderr)
        bad = m and m.group(1).replace("/", ".")
        if not bad or bad not in mods:
            sys.exit(f"ImportAudit.lean failed:\n{out.stderr[-2000:]}")
        mods.remove(bad)
        stale.append(bad)
    req = {}
    for line in out.stdout.splitlines():
        m, _, rest = line.partition("\t")
        req[m] = set(rest.split())
    return req


def compiles_without(path, imp):
    """True/False: does `path` compile with the import(s) `imp` (a name or a set) removed?
    None: not tried (a top-level `#eval` would run)."""
    drop = {imp} if isinstance(imp, str) else set(imp)
    text = open(path).read()
    if re.search(r"^#eval\b", text, re.M):
        return None
    lines = [l for l in text.split("\n") if l.strip() not in {f"import {d}" for d in drop}]
    with tempfile.NamedTemporaryFile("w", suffix=".lean", dir=tempfile.gettempdir(),
                                     delete=False) as fh:
        fh.write("\n".join(lines))
        tmp = fh.name
    try:
        r = subprocess.run(["lake", "env", "lean", tmp], capture_output=True, text=True)
        return r.returncode == 0
    finally:
        os.unlink(tmp)


def lib_reachable():
    """Modules some lean_lib builds: every root the lakefile names, and their import closures
    (the `Apps` lib is its globs, so apps/ and demos/ count whole)."""
    text = open("lakefile.lean").read()
    roots = set(re.findall(r"`(LeanMlir(?:\.[A-Za-z0-9_]+)*)", text))
    reach = set(roots)
    for r in roots:
        reach |= closure(r)
    return reach


def unused(verify):
    files = [f for f in project_files(False) if f != "LeanMlir.lean"]
    reach = lib_reachable()
    files = [f for f in files if not f.startswith("LeanMlir/") or module_of(f) in reach]
    built = [f for f in files if os.path.exists(olean(module_of(f)))]
    skipped = sorted(set(files) - set(built))
    stale = []
    # the library modules load into one environment; each app/demo defines its own `main`, so
    # those load one per run
    lib = [module_of(f) for f in built if f.startswith("LeanMlir/")]
    exes = [module_of(f) for f in built if not f.startswith("LeanMlir/")]
    req = required_modules(lib, stale)
    with ThreadPoolExecutor(max_workers=max(1, (os.cpu_count() or 4) // 2)) as ex:
        for r in ex.map(lambda m: required_modules([m], stale), exes):
            req.update(r)
    built = [f for f in built if module_of(f) not in stale]
    cands = []
    for f in built:
        m = module_of(f)
        imps = imports_of(m)
        for i in imps:
            if any(i in closure(y) for y in imps if y != i):
                continue                                   # implied — `implied` reports it
            others = set()
            for j in imps:
                if j != i:
                    others |= {j} | closure(j)
            if req.get(m, set()) <= others:
                cands.append((f, i))
    if skipped:
        print(f"skipped (in a lib but not built yet — run `lake build` first): {' '.join(skipped)}\n")
    if stale:
        print(f"skipped (unreadable .olean — an old toolchain's): {' '.join(stale)}\n")
    if not verify:
        for f, i in cands:
            print(f"{f}: {i}")
        print(f"\n{len(cands)} candidate(s) by constants; confirm with --verify")
        return 0
    with ThreadPoolExecutor(max_workers=max(1, (os.cpu_count() or 4) // 2)) as ex:
        results = list(ex.map(lambda c: compiles_without(*c), cands))
    confirmed = [c for c, ok in zip(cands, results) if ok]
    unrun = [c for c, ok in zip(cands, results) if ok is None]
    # each was confirmed ALONE; where a file has several, they may only be removable one at a
    # time (two imports each supplying the same thing) — compile once with all of them gone
    per_file = {}
    for f, i in confirmed:
        per_file.setdefault(f, []).append(i)
    either = {f for f, imps in per_file.items()
              if len(imps) > 1 and not compiles_without(f, imps)}
    for f, i in confirmed:
        note = "  (either-or: removable alone, not together with the others listed)" if f in either else ""
        print(f"{f}: {i}{note}")
    for f, i in unrun:
        print(f"{f}: {i}  (not compiled: the file has a top-level #eval)")
    print(f"\n{len(confirmed)} unused import(s) confirmed by compiling; "
          f"{len(cands) - len(confirmed) - len(unrun)} candidate(s) needed after all "
          "(notation, tactics, #eval, example); "
          f"{len(unrun)} not compiled")
    return 0


def main():
    args = sys.argv[1:]
    if not args or args[0] not in ("implied", "unused"):
        sys.exit(__doc__)
    if args[0] == "implied":
        return implied("--all" in args)
    return unused("--verify" in args)


if __name__ == "__main__":
    sys.exit(main())
