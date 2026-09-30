#!/usr/bin/env python3
"""Every module a Lean file cites by name or path is a module that exists.

`lake exe docstring-checkrefs` resolves the `Ident`s a docstring cites against the environment,
but only the ones carrying a project marker (`HasVJP`, `_correct`, `Render`, …), and it checks a
`dir/File.lean` span only when doc-gen4 would link it. Module citations fall through both:
`Foundation.DataParallelSyncBf16` and `Smoothing.DecChunk1` carry no marker, and a path written
in prose or in an emitted string (`in LeanMlir/Proofs/Foundation/DataParallelSyncBf16.lean`) is
not a backticked span. A move or rename leaves them pointing at nothing with every build green.

Two checks, over every tracked `.lean` file (comments, docstrings and string literals alike):

* **Paths.** A `…/File.lean` token must be a tracked file: the full repo path, or a path suffix
  of one (`Foundation/DataParallel/SyncBf16.lean`). Upstream trees (`Mathlib/`, `Lean/`, …),
  globs and placeholders (`*`, `{`, `<`) are skipped.
* **Dotted module names.** A token `A.B…` whose components all start upper-case and whose first
  component is a directory of this repo's Lean tree (`Foundation`, `Smoothing`, `Nets`, …) or
  `LeanMlir` must end-match a module (`Foundation.DataParallel.SyncBf16`) or a namespace or
  declaration the sources declare (`Proofs.StableHLO.SHlo`). A token whose last component is
  lower-case is a declaration reference, which is docstring-checkrefs' job.

Toolchain-free (Python and git), so it runs in the unfiltered targets job.
`module_refs_allow.tsv` (`file<TAB>token<TAB>why`) holds known misses whose fix is deferred;
it may shrink, never grow.

    python3 scripts/gates/module_refs.py       # exit 1 and list every miss
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ("Mathlib/", "Lean/", "Init/", "Std/", "Lake/", "Batteries/", "Aesop/", "Qq/",
            "Plausible/", "ProofWidgets/", "DocGen4/", "Verso/")
PATH = re.compile(r"[\w.\-/*{}<>…]*/[\w\-*{}<>…]+\.lean\b")
DOTTED = re.compile(r"(?<![\w.])([A-Z][\w']*(?:\.[A-Z][\w']*)+)(?![\w'])")
PLACEHOLDER_STEMS = {"File", "X", "Foo", "Bar"}
ALLOW = Path(__file__).resolve().parent / "module_refs_allow.tsv"
NAMESPACE = re.compile(r"^\s*(namespace|section|end)\b[ \t]*([^\s\-]*)", re.M)
DECL = re.compile(
    r"^\s*(?:@\[[^\]]*\]\s*)*(?:(?:private|protected|noncomputable|partial|unsafe|nonrec)\s+)*"
    r"(?:theorem|lemma|def|abbrev|instance|structure|inductive|class|opaque|axiom)\s+"
    r"([^\s:({\[]+)", re.M)


def tracked_lean() -> list[str]:
    out = subprocess.run(["git", "ls-files", "*.lean"], cwd=ROOT, capture_output=True,
                         text=True, check=True).stdout.split("\n")
    return [p for p in out if p and not p.startswith(".lake/") and (ROOT / p).exists()]


SCAN_SKIP = ("historical/", "planning/")   # archives and drafts: cited, not citing


def declared_names(text: str) -> tuple[set[tuple[str, ...]], set[tuple[str, ...]]]:
    """Namespaces and declaration names a file declares, as component tuples. A plain text
    walk of `namespace`/`section`/`end` and the declaration keywords: enough for the
    upper-case names this gate resolves. Returns `(namespaces, declarations)`."""
    events = [(m.start(), "ns", m.group(1), m.group(2)) for m in NAMESPACE.finditer(text)]
    events += [(m.start(), "decl", "", m.group(1)) for m in DECL.finditer(text)]
    events.sort()
    stack: list[tuple[str, list[str]]] = []
    out: set[tuple[str, ...]] = set()
    decls: set[tuple[str, ...]] = set()
    cur: list[str] = []
    for _, kind, kw, name in events:
        if kind == "ns":
            if kw == "namespace":
                comps = name.split(".")
                stack.append(("namespace", comps))
                cur = cur + comps
                out.add(tuple(cur))
            elif kw == "section":
                stack.append(("section", []))
            elif stack:
                _, comps = stack.pop()
                cur = cur[:len(cur) - len(comps)] if comps else cur
        else:
            name = name.removeprefix("_root_.")
            decls.add(tuple(cur + name.split(".")))
    return out, decls


def main() -> int:
    files = tracked_lean()
    paths = [tuple(p.split("/")) for p in files]
    modules = {tuple(p[:-len(".lean")].split("/")) for p in files}
    dirs = {c for p in paths for c in p[:-1]} | {"LeanMlir"}
    names: set[tuple[str, ...]] = set()
    decls: set[tuple[str, ...]] = set()
    texts = {}
    for f in files:
        texts[f] = (ROOT / f).read_text(errors="replace")
        ns, ds = declared_names(texts[f])
        names |= ns | ds
        decls |= ds
    # a module directory is a name too (`LeanMlir.Proofs.Nets`)
    known = modules | names | {m[:i] for m in modules for i in range(1, len(m))}
    allow = set()
    if ALLOW.exists():
        for row in ALLOW.read_text().split("\n"):
            if row.strip() and not row.startswith("#"):
                f, tok, *_ = row.split("\t")
                allow.add((f, tok))

    def path_ok(tok: str) -> bool:
        tok = tok.lstrip("./")
        if any(c in tok for c in "*{}<>…") or tok.startswith(UPSTREAM) or ".lake/" in tok:
            return True
        if Path(tok).stem in PLACEHOLDER_STEMS:     # `dir/File.lean`, `tests/X.lean`
            return True
        parts = tuple(tok.split("/"))
        return any(p[len(p) - len(parts):] == parts for p in paths if len(p) >= len(parts))

    def dotted_ok(tok: str) -> bool:
        parts = tuple(tok.split("."))

        def ends(pool, ps):
            return any(k[len(k) - len(ps):] == ps for k in pool if len(k) >= len(ps))
        # a field of a declared structure (`ViTTieWeights.Wc`) is cited through its prefix
        return ends(known, parts) or ends(decls, parts[:-1])

    misses = []
    for f in files:
        if f.startswith(SCAN_SKIP):
            continue
        for n, line in enumerate(texts[f].split("\n"), 1):
            if line.startswith("import "):
                continue                      # Lean itself resolves these
            for m in PATH.finditer(line):
                tok = m.group(0)
                if tok.startswith(("http", "//")) or "github.com" in line[:m.start()][-80:]:
                    # a repo link's target is docstring-checkrefs' file-citation check
                    continue
                if not path_ok(tok) and (f, tok) not in allow:
                    misses.append((f, n, tok, "no tracked file at this path"))
            for m in DOTTED.finditer(line):
                tok = m.group(1)
                if tok.split(".")[0] not in dirs:
                    continue
                if not dotted_ok(tok) and (f, tok) not in allow:
                    misses.append((f, n, tok, "no module, namespace or declaration by this name"))
    if misses:
        print(f"{len(misses)} module citation(s) resolve to nothing:")
        for f, n, tok, why in misses:
            print(f"✗ {f}:{n}: `{tok}` — {why}")
        return 1
    print("✓ every module path and dotted module name cited in a Lean file resolves")
    return 0


if __name__ == "__main__":
    sys.exit(main())
