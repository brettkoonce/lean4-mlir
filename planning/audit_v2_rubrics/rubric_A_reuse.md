# Part A — reuse: what should come from Mathlib (or core/Batteries, or one in-repo kit)

## What counts as a finding

In descending order of value:

1. **A declaration Mathlib / core / Batteries already has.** A definition or lemma that an existing
   library declaration directly replaces. Highest value: it deletes code.
2. **A proof re-deriving a standard result.** Any proof more than a few lines whose goal, or a key
   intermediate step, is a named library lemma. Standard plumbing — image/preimage, `Finset`/`Finsupp`
   support arithmetic, `Fin` case bashes, `List`/`Vector`/`Array` index juggling, monotonicity and
   continuity/differentiability side conditions, `fderiv` of affine/linear maps — almost always has one.
3. **A definition assembled from raw pieces** where a library combinator does the assembly.
4. **A hand-rolled special case of a general library result** — fixed dimension, concrete type, one
   instance.
5. **Near-clones inside this repo.** Two or more places with the same proof shape or distinctive
   identifiers, which should be one shared construction — including code written since v1 that
   re-derives what an existing repo kit already provides. Report even without a Mathlib replacement:
   the fix is factoring. Give the sites (file:line each) and the shared statement.

For non-proof Lean (codegen, runtime, parsers): the same question against core `Std`/`Lean` —
`String`/`Array`/`List`/`HashMap` utilities, `Nat.toDigits`, formatting, parsing.

## How to search

For each declaration in scope, actually run searches — do not judge from memory:

```bash
grep -rn "theorem <guess>\|lemma <guess>\|def <guess>" .lake/packages/mathlib/Mathlib | head
grep -rn "Continuous.*comp\|deriv.*add" .lake/packages/mathlib/Mathlib --include=*.lean | head
grep -rn "<identifier>" --include=*.lean LeanMlir tests apps demos
```

In a scratch file use `exact?`, `apply?`, `#check`, and `#leansearch "<description>"`
(`LeanSearchClient` is a dependency; it needs network — if it fails, fall back to grep).
Typechecking is stronger evidence than grep.

Keep searching until you find the replacement **or are confident it is absent**. A finding you cannot
name a replacement for is not a finding (except category 5, where the replacement is the shared
statement you write down).

## Evidence standard

Every finding names the located replacement and shows exactly how to use it: the fully qualified
name, the file it lives in, and a one-liner demonstrating the substitution — **typechecked in a
scratch file** for `verified`. Say how many consumers the repo declaration has (so the reader knows
whether it is a delete or a bridge).

## What is NOT a finding

- Mathlib itself restates generic lemmas per type. A specialization with genuine consumers here earns
  its place — check whether it is used before flagging.
- Similar-looking but different in a way that matters (hypotheses, conclusion, definitional
  unfolding). Say what you checked, under "Checked, not findings".
- Generated files, `.lake/`, `.venv*/`, vendored deps. Style/naming/formatting (Part B covers
  structure, not style). Anything needing a Mathlib newer than `v4.34.0`.

## Finding format

```
### <path>:<line> — <the duplicated declaration>

**Replace with:** `Fully.Qualified.name` (`.lake/packages/mathlib/Mathlib/<path>`)
**Why it matches:** <same statement / up to renaming / special case of — what you verified>
**How:** <one-liner, or import + call>
**Consumers / lines saved:** <n uses; ~k lines>
**Confidence:** verified | likely | lead
```
