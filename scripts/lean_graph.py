"""The Lean module graph as the lakefile and the sources state it — pure text, no Lean, no build.

`lib_roots` reads one `lean_lib`'s `roots := #[...]`; `reachable` walks `import` lines from a set
of roots. Shared by `check_audit_coverage.py` and `audit_census/run.sh`.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LAKEFILE = ROOT / "lakefile.lean"


def lakefile_text():
    return LAKEFILE.read_text()


def libs(text):
    """Every `lean_lib` name, in lakefile order (e.g. `«Certs»`)."""
    return re.findall(r"^lean_lib\s+(\S+)\s+where", text, re.M)


def lib_roots(text, lib):
    """Roots of one lib: the names inside its `roots := #[...]` only.

    `--` comments are stripped first (they hold brackets like [3,4,6,3] that end a naive match
    early), and the scan stops at the array's `]` — not at the next `lean_lib`, which would also
    sweep in the next lib's docstring and count every backticked name there as a root.
    """
    try:
        segment = text.split(f"lean_lib {lib} where", 1)[1].split("lean_lib", 1)[0]
    except IndexError:
        sys.exit(f"error: no `lean_lib {lib}` in lakefile.lean")
    code = "\n".join(line.split("--", 1)[0] for line in segment.splitlines())
    m = re.search(r"roots\s*:=\s*#\[(.*?)\]", code, re.S)
    if not m:
        sys.exit(f"error: `lean_lib {lib}` has no `roots := #[...]`")
    return re.findall(r"`([A-Za-z0-9_.]+)", m.group(1))


def source_of(module):
    return ROOT / (module.replace(".", "/") + ".lean")


def imports_of(module):
    """The `import`s of `module`'s source, or None when it has no source file."""
    path = source_of(module)
    if not path.exists():
        return None
    return re.findall(r"^import\s+([A-Za-z0-9_.]+)", path.read_text(), re.M)


def reachable(roots, prefix="LeanMlir", strict=True):
    """Every `prefix` module reachable from `roots` through `import`. `strict` exits on a module
    with no source file; otherwise it is skipped."""
    seen = set()
    stack = [r for r in roots if r.startswith(prefix)]
    while stack:
        module = stack.pop()
        if module in seen:
            continue
        imps = imports_of(module)
        if imps is None:
            if strict:
                sys.exit(f"error: root/import `{module}` has no source file at {source_of(module)}")
            continue
        seen.add(module)
        stack += [m for m in imps if m.startswith(prefix)]
    return seen
