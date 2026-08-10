#!/usr/bin/env python3
"""Which certified forwards have NO backward-graph faithfulness theorem?

§8e of `planning/mnv4_verified.md` ran this sweep by hand and **took four passes, three of
which were wrong**, because Lean naming does not support mechanical auditing:

  pass 1 "151 holes"  — matched `<X>_has_vjp` against `<X>Back*Graph_faithful`; swept up
                        whole-net forwards, primitives, weight-grads and the float-bridge world
  pass 2 "6 holes"    — narrowed to batched `*B`, but missed the `<X>GraphB_faithful` family
                        entirely (no "Back" in the name)
  pass 3 "16 holes"   — ...those are FORWARD-graph theorems (`den (G e) = fwd (den e)`), so the
                        discriminator has to be "does the statement mention `.backward`"
  pass 4 "5 holes" ✅  — ...and the capture regex terminated at the first `:=`, which occurs
                        INSIDE statements as named arguments `(h := h)`

It was never committed, so the next session re-derives it and re-makes those mistakes. This is
pass 4, written down. The statement/proof split is by **paren depth**: a named-argument `:=`
is always inside parentheses, so the first depth-0 `:=` is the real boundary.

⚠ **A name-keyed audit cannot tell one object with two names from two objects.** §8f found
`mbStridedFwdB_has_vjp` and `mbDownBodyB_has_vjp` are definitionally the same VJP (`rfl`), so a
reported hole can be a naming artifact — two of §8e's four were. Probe before building.

Two cohorts, because the repo has two VJP worlds:
  * **batched** `Vec`-level `<X>B_has_vjp` — the conv nets (§8e's scope)
  * **per-token** `Mat`-level `<X>_has_vjp_mat` — ViT

Usage:  python3 scripts/vjp_graph_sweep.py [--all] [--cohort batched|mat|both]
        --all  also lists the TIED forwards, not just the holes
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PROOFS = ROOT / "LeanMlir" / "Proofs"

# A declaration header at column 0, optionally `private`/`protected`/`noncomputable`.
DECL_RE = re.compile(
    r"^(?:private\s+|protected\s+|noncomputable\s+)*"
    r"(theorem|lemma|def|abbrev|instance)\s+([A-Za-z_][A-Za-z0-9_.'!?]*)",
)


def split_statement(body: str) -> str:
    """Statement text = up to the first `:=` (or `where`) at paren/bracket depth 0.

    ⭐ This is pass 4's fix. `(h := h)` and `(Np1 := 197)` are named arguments and live at
    depth > 0; the proof-introducing `:=` is the first one at depth 0.
    """
    depth = 0
    i = 0
    n = len(body)
    while i < n:
        c = body[i]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif depth == 0:
            if body.startswith(":=", i):
                return body[:i]
            if body.startswith("where", i) and (i + 5 >= n or not body[i + 5].isalnum()):
                return body[:i]
        i += 1
    return body


def parse_decls(path: pathlib.Path) -> list[tuple[str, str, str]]:
    """-> [(name, statement, full_body)] for every top-level declaration in `path`."""
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    starts: list[tuple[int, str]] = []
    for idx, line in enumerate(lines):
        m = DECL_RE.match(line)
        if m:
            starts.append((idx, m.group(2)))
    out = []
    for j, (idx, name) in enumerate(starts):
        end = starts[j + 1][0] if j + 1 < len(starts) else len(lines)
        body = "\n".join(lines[idx:end])
        out.append((name, split_statement(body), body))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="also list tied forwards")
    ap.add_argument("--cohort", choices=["batched", "mat", "both"], default="batched")
    args = ap.parse_args()

    decls: list[tuple[str, str, str, pathlib.Path]] = []
    for path in sorted(PROOFS.rglob("*.lean")):
        for name, stmt, body in parse_decls(path):
            decls.append((name, stmt, body, path))

    # ── The two forward cohorts ──────────────────────────────────────────────
    # batched: `<X>B_has_vjp` / `<X>B_has_vjp_at` — the trailing B before `_has_vjp` is what
    # `*BackB0` uses for "batched over N", and it is what keeps whole-net and per-example
    # forwards (pass 1's 151) out of scope.
    batched = {}
    mat = {}
    for name, stmt, _body, path in decls:
        m = re.fullmatch(r"(.*B)_has_vjp(?:_at)?", name)
        if m:
            batched.setdefault(m.group(1), (name, path))
        m = re.fullmatch(r"(.*)_has_vjp_mat", name)
        if m:
            mat.setdefault(m.group(1), (name, path))

    # ── Backward-graph faithfulness theorems ─────────────────────────────────
    # ⭐ pass 3's discriminator: the STATEMENT must mention `.backward`. A `<X>GraphB_faithful`
    # is a FORWARD-graph theorem (`den (G e) = fwd (den e)`) and must not count.
    # ⚠ The name filter must include `*_eq_backward`, not just `*_faithful`/`*_den`: ViT's
    # per-token token bridges (`rowDenseBackFlat_eq_backward`, `geluFlat_eq_backward`,
    # `rowVecLNBack_eq_backward`, `patchEmbedBackFlat_eq_backward`) make exactly the same claim
    # — the emitted backward token denotes the proven VJP backward — under a different naming
    # convention. Omitting them reports the whole per-token primitive layer as uncovered.
    back_stmts: list[str] = []
    back_bodies: list[str] = []
    for name, stmt, body, _path in decls:
        if not ("_faithful" in name or name.endswith("_den") or name.endswith("_eq_backward")):
            continue
        if "den" not in stmt or ".backward" not in stmt:
            continue
        back_stmts.append(stmt)
        back_bodies.append(body)
    stmt_blob = "\n".join(back_stmts)
    body_blob = "\n".join(back_bodies)

    # ⚠⚠ The ONLY calibrated cohort is `batched`: its verdict was derived by hand in §8e/§8f and
    # this script reproduces it (1 hole, `efficientnetForwardB`). `mat` has NEVER been checked
    # against a hand-derived answer, and its classification is known to be unreliable —
    # `mhsa_g`/`colSlabwise` are demonstrably consumed on the way to `mhsaBackGraphMH_faithful`
    # (via `mhsaClean`, a def, and `mhsaClean_backward_collapse`, whose name matches no filter)
    # yet land in the hole column. Do NOT read that column as debt; tuning the filter until it
    # looks clean is how a sweep manufactures a false green.
    EXPECTED_BATCHED_HOLES = {"efficientnetForwardB"}

    rc = 0
    for cohort_name, cohort in (("batched", batched), ("mat", mat)):
        if args.cohort not in (cohort_name, "both"):
            continue
        if cohort_name == "mat":
            print("⚠ UNCALIBRATED COHORT — the columns below have never been checked against a")
            print("  hand-derived answer. Ingredients-of-ingredients land in the hole column.\n")
        # ⭐ THREE categories, not two. A forward whose VJP is consumed INSIDE a capstone's proof
        # (an `mhsa_g_has_vjp_mat` fed to `colSlabwise_has_vjp_mat` on the way to
        # `mhsaBackGraphMH_faithful`) is an INGREDIENT of a certified graph, not a stage nobody
        # covered. Collapsing those into "hole" is how a sweep manufactures debt: the first run
        # of this script reported 9 per-token holes, of which several were ingredients.
        holes, tied, ingredients = [], [], []
        for stem, (vjp_name, path) in sorted(cohort.items()):
            if vjp_name in stmt_blob:
                tied.append((stem, vjp_name, path))
            elif vjp_name in body_blob:
                ingredients.append((stem, vjp_name, path))
            else:
                holes.append((stem, vjp_name, path))
        total = len(cohort)
        print(f"── {cohort_name}: {total} certified forwards — {len(tied)} tied, "
              f"{len(ingredients)} ingredient-only, {len(holes)} with NO backward graph")
        for stem, vjp_name, path in holes:
            print(f"   ⛔ {stem:<34} ({vjp_name}, {path.relative_to(ROOT)})")
        for stem, _vjp_name, _path in ingredients:
            print(f"   ◦  {stem:<34} (consumed inside a capstone's proof, not a stage hole)")
        if args.all:
            for stem, _vjp_name, _path in tied:
                print(f"   ✓  {stem}")
        print()
        # ⭐ THE RATCHET, and it can genuinely fail: the batched hole set is pinned to the
        # hand-verified §8f answer, so a change that opens a new hole (or silently closes the
        # known one without updating the ledger) is a non-zero exit, not a prettier number.
        if cohort_name == "batched":
            got = {stem for stem, _n, _p in holes}
            if got != EXPECTED_BATCHED_HOLES:
                print(f"⛔ RATCHET FAILED — batched holes {sorted(got)} "
                      f"≠ ledger {sorted(EXPECTED_BATCHED_HOLES)}")
                rc = max(rc, 1)
            else:
                print("✅ batched cohort matches the §8f ledger "
                      "(the one hole is the whole-net forward, = the artifact-tie item)")

    print("⚠ A hole here can be a NAMING artifact — two names for one `rfl`-equal VJP look")
    print("  like two objects to any name-keyed sweep (§8f: 2 of 4 were). Probe before building.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
