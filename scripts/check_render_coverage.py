#!/usr/bin/env python3
"""Assert every committed-artifact renderer is BUILT by CI and GUARDED against drift.

The sibling of scripts/check_audit_coverage.py, for the other half of the
Proofs/Codegen surface. Two invariants, both of which have silently broken:

  (1) BUILD REACHABILITY. A renderer that nothing imports is reachable only as
      a `lean_lib` root. `lean_lib «Certs»`'s docstring claims its roots
      "subsume the `Proofs` roots above, so building `Certs` builds
      everything" — that claim was FALSE for ResNet34RenderB and
      MobileNetV2RenderB, which are leaf `#eval` writers with zero importers.
      On a fresh runner `lake build Certs` never produced their oleans, and any
      job needing one died with

        uncaught exception: object file '….../ResNet34RenderB.olean' of module
        LeanMlir.Proofs.Codegen.ResNet34RenderB does not exist

      Locally the gap hides behind stale oleans from dev builds — this is
      exactly the failure that only shows up in CI.

  (2) DRIFT GUARD. proofs.yml re-elaborates each renderer and `git diff
      --exit-code`s its artifact, so a renderer edit that changes the emitted
      IR fails the push. A writer missing from that list is a writer whose
      artifact can drift undetected — and per the §2n lesson, a green build
      plus an empty artifact diff is VACUOUS when the `#eval` never ran.

Both are pure text analysis: parse the lakefile roots, BFS the import graph,
scan for artifact-writing `#eval`s, and read the workflow. No Lean, no build,
so it fails fast as the first CI step.

Invariant (1) is a hard failure — it is the one that breaks the build, and it
currently holds. Invariant (2) is enforced against a BASELINE
(scripts/render_guard_baseline.txt) of artifacts known to be unguarded today,
because closing that debt means re-elaborating the heavy batched renderers on
every push and is a CI-budget decision, not a correctness one. Anything not in
the baseline fails: the debt can shrink but never grow. Delete a line from the
baseline once its artifact is diffed by the guard.

Usage: python3 scripts/check_render_coverage.py [--update-baseline]
       (from the repo root)
"""

import re
import sys
from fnmatch import fnmatch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_audit_coverage import reachable, lib_roots  # noqa: E402

LAKEFILE = Path("lakefile.lean")
CODEGEN = Path("LeanMlir/Proofs/Codegen")
WORKFLOW = Path(".github/workflows/proofs.yml")
BASELINE = Path("scripts/render_guard_baseline.txt")

# Writers that emit to scratch space rather than the committed tree: they are
# demonstrations, not artifacts, so neither invariant applies.
SCRATCH_ONLY = {"IRPrint"}


def artifact_writers() -> dict[str, list[str]]:
    """Map module basename -> the committed artifact paths its `#eval`s write.

    Matches `#eval IO.FS.writeFile "verified_mlir/…"`, allowing the path to sit
    on the following line (several renderers wrap it that way).

    ⚠ ALSO matches the `#eval do … IO.FS.writeFile "…"` block form. That form was
    invisible here until 2026-08-26, which is this checker's own failure mode: three
    committed artifacts (`cifar8{_bf16,b,wb}_fwd.mlir`) were written by `#eval do`
    blocks, so they were neither drift-guarded NOR baselined and the check still
    reported OK. A writer the scanner cannot see is exactly the gap the scanner
    exists to close, so the window is bounded rather than unbounded: only writes
    within WINDOW chars of an `#eval` count, which keeps helper `def`s that write
    to scratch paths from being mistaken for elaboration-time artifact writers.
    """
    WINDOW = 400
    out: dict[str, list[str]] = {}
    for path in sorted(CODEGEN.glob("*.lean")):
        if path.stem in SCRATCH_ONLY:
            continue
        text = path.read_text()
        # collapse the `#eval IO.FS.writeFile\n  "path"` wrap before matching
        flat = re.sub(r"\s+", " ", text)
        hits = [m.group(1)
                for ev in re.finditer(r'#eval\b', flat)
                for m in re.finditer(r'IO\.FS\.writeFile\s+"([^"]+)"',
                                     flat[ev.start(): ev.start() + WINDOW])]
        committed = [h for h in hits if not h.startswith("/tmp")]
        if committed:
            out[path.stem] = sorted(set(committed))
    return out


def expand_braces(pattern: str) -> list[str]:
    """Expand one level of shell brace alternation: a{b,c}d -> [abd, acd].

    The drift guard writes its paths the way you would type them at a shell —
    `verified_mlir/{linear,mlp,cnn}_*.mlir` — so a literal string match sees
    none of them and would report guarded artifacts as unguarded.
    """
    m = re.search(r"\{([^{}]*)\}", pattern)
    if not m:
        return [pattern]
    out = []
    for alt in m.group(1).split(","):
        out += expand_braces(pattern[:m.start()] + alt + pattern[m.end():])
    return out


def guarded_patterns(workflow_text: str) -> list[str]:
    """Every verified_mlir path the workflow diffs, brace-expanded.

    Scoped to `git diff` lines so a path merely NAMED in an error message or a
    comment does not count as guarded.
    """
    patterns: list[str] = []
    for line in workflow_text.splitlines():
        if "git diff" not in line and not line.strip().startswith("verified_mlir/"):
            continue
        for tok in re.findall(r"verified_mlir/[^\s\"'|]+", line):
            patterns += expand_braces(tok)
    return patterns


def main() -> None:
    if not LAKEFILE.exists():
        sys.exit("error: run me from the repo root (no lakefile.lean here)")
    text = LAKEFILE.read_text()
    writers = artifact_writers()
    if not writers:
        sys.exit(f"error: found no artifact writers under {CODEGEN} — "
                 f"the detection regex has probably rotted")

    proofs = reachable(lib_roots(text, "«Proofs»"))
    certs = reachable(lib_roots(text, "«Certs»"))
    guard = WORKFLOW.read_text() if WORKFLOW.exists() else ""
    patterns = guarded_patterns(guard)

    unbuilt: list[tuple[str, str]] = []
    unguarded: list[tuple[str, list[str]]] = []
    for stem, artifacts in writers.items():
        module = f"LeanMlir.Proofs.Codegen.{stem}"
        for lib, covered in (("Proofs", proofs), ("Certs", certs)):
            if module not in covered:
                unbuilt.append((module, lib))
        # the guard must BOTH re-elaborate the renderer and diff its artifacts
        elaborated = f"Codegen/{stem}.lean" in guard
        # An artifact is guarded only if the renderer is re-elaborated AND the
        # artifact is diffed: diffing a file nobody regenerated proves nothing.
        missing_art = artifacts if not elaborated else [
            a for a in artifacts if not any(fnmatch(a, p) for p in patterns)]
        if missing_art:
            unguarded.append((stem, missing_art, elaborated))

    # Invariant (2) is enforced against the baseline: an unguarded artifact is
    # a failure unless it is already recorded as known debt.
    known = set()
    if BASELINE.exists():
        known = {ln.strip() for ln in BASELINE.read_text().splitlines()
                 if ln.strip() and not ln.startswith("#")}
    current = sorted({a for _stem, missing, _e in unguarded for a in missing})
    if "--update-baseline" in sys.argv:
        BASELINE.write_text(
            "# Artifacts written by a Proofs/Codegen renderer that the proofs.yml\n"
            "# drift guard does NOT diff. Generated by\n"
            "#   python3 scripts/check_render_coverage.py --update-baseline\n"
            "# This list may SHRINK, never grow: a new entry means a renderer's\n"
            "# output can drift with CI green. Delete a line once the guard diffs it.\n"
            + "".join(f"{a}\n" for a in current))
        print(f"baseline updated: {len(current)} unguarded artifact(s) recorded "
              f"in {BASELINE}")
        return
    new_debt = [a for a in current if a not in known]
    stale = sorted(known - set(current))

    rc = 0
    if unbuilt:
        print("error: artifact renderer(s) not reachable from a lib root — "
              "their .oleans will not exist on a fresh runner:", file=sys.stderr)
        for module, lib in unbuilt:
            print(f"  {module}  (missing from `{lib}`)", file=sys.stderr)
        print("\nfix: add the module to that lean_lib's `roots` in lakefile.lean. "
              "Nothing imports a leaf renderer, so being a root is the ONLY way "
              "it gets built.", file=sys.stderr)
        rc = 1
    if new_debt:
        print("\nerror: artifact(s) newly unguarded by the drift guard in "
              f"{WORKFLOW} (not in {BASELINE}):", file=sys.stderr)
        for a in new_debt:
            owner = next(st for st, miss, _e in unguarded if a in miss)
            print(f"  {a}  (written by {owner})", file=sys.stderr)
        print("\nfix: add to the 'Verified-render drift guard' step:\n"
              "  lake env lean LeanMlir/Proofs/Codegen/<Module>.lean\n"
              "  git diff --exit-code -- <artifact> || { echo '::error::…'; exit 1; }\n"
              "Only if the gap is deliberate and budgeted: re-record it with\n"
              "  python3 scripts/check_render_coverage.py --update-baseline",
              file=sys.stderr)
        rc = 1
    if rc:
        sys.exit(rc)

    n_art = sum(len(a) for a in writers.values())
    print(f"render coverage OK: {len(writers)} artifact renderer(s) writing "
          f"{n_art} committed file(s) — all reachable from BOTH the Proofs and "
          f"Certs roots.")
    if current:
        print(f"  drift guard: {n_art - len(current)}/{n_art} artifacts diffed; "
              f"{len(current)} known-unguarded (baselined, see {BASELINE})")
        for stem, missing_art, elaborated in unguarded:
            if not elaborated:
                print(f"    ! {stem} is never re-elaborated — writes "
                      f"{len(writers[stem])} artifact(s) incl. "
                      f"{writers[stem][0]}")
    if stale:
        print(f"  note: {len(stale)} baseline entr(y/ies) now guarded — "
              f"prune them from {BASELINE}:")
        for a in stale:
            print(f"    {a}")


if __name__ == "__main__":
    main()
