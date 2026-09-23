# Part B — clarity: is this code easy to understand, and are its boundaries in the right places?

Part A asks "does this code already exist elsewhere?". Part B asks a different question: **could a
competent Lean + ML reader, new to this repo, open this slice and understand what it proves, where
each concept lives, and which file to change for a given edit?** You are looking for structure that
makes that hard — not for style, formatting, or individual naming nits.

Target reader: knows Lean 4, Mathlib, and backprop; has read the README and the book's chapter for
the net; has NOT read the git history or `planning/`.

## What counts as a finding

In roughly descending order of value:

1. **Misplaced ownership.** A concept, definition, or general lemma living in a consumer file instead
   of the module that owns the concept (e.g. a generic `Finset` reindexing lemma parked in an SGD
   file; a BN fact living in a ResNet file). Name the owner it belongs in, check the move creates no
   import cycle, and count importers of the destination (the rebuild cost).
2. **A file or namespace without one job.** A file mixing layers that change for different reasons —
   spec + VJP + codegen tie + float bound + seal in one place — or a "god file" whose sections a
   reader cannot navigate. Propose the seam: which declarations go where, and why that cut is the
   natural one (what depends on what). Don't propose splits for size alone.
3. **Leaky abstractions.** Consumers reaching through an abstraction instead of using its API:
   `unfold X` / `simp [X]` / `delta` / `show <unfolded X>` / `change` on a definition owned by another
   module, repeated across files; proofs relying on accidental defeq across a module boundary; a kit
   (`CertLayer`, `HasVJP`, clause Props, `FloatClose`) whose consumers routinely re-open its fields.
   Count sites with grep; propose the missing API lemma(s) that would seal it.
4. **One concept, several spellings.** The same mathematical object defined twice under different
   names or representations (per-example vs batched vs flat vs `Mat`; `Fin (a*b)` vs `Fin a × Fin b`),
   with bridges scattered around. Or parallel hierarchies whose correspondence a reader must
   reverse-engineer. Propose the canonical spelling and where the bridges should live (one place).
5. **Opaque naming schemes.** Not individual names — *schemes*: suffix soups (`B`, `B0`, `GB`, `PoC`,
   `PoCB`, `PoCG`, `PC`, `Live`, `Realistic`, `V`, `2`, …) whose meaning is historical rather than
   descriptive, or names that encode an obsolete plan. Report the scheme, what each suffix actually
   means today (determine it from the code), whether it is used consistently, and whether it is
   documented anywhere a reader would find it. Propose either a legend (cheap) or a rename (cost it:
   pinned names, book, yaml — see shared facts).
6. **Missing entry points.** A file/directory where a reader can't tell what the top theorem is,
   what the file proves, or how it fits the chain (module docstring missing, stale, or describing
   history instead of content). Stale docstrings that cite things that no longer exist or describe
   a superseded design count here — give file:line and what is wrong.
7. **Dependency direction and weight.** Imports that point the wrong way (a Foundation module
   importing a net; a spec module importing codegen), or a heavy import pulled in for one small
   lemma. Use the `import` lines; `lake env lean --deps`-style tooling is not needed.
8. **Indirection that explains nothing.** Wrapper definitions/lemmas that only forward to another
   with no change of meaning, abbreviation layers a reader must peel, `let`-chains or local
   notation that hide what a statement says. (Contrast: a named clause Prop that makes a 16M-heartbeat
   statement readable is GOOD indirection — don't flag it.)

Also note **good patterns** worth copying elsewhere in the repo (one line each, under "Checked, not
findings") — e.g. a file whose docstring + layout is the model the others should follow.

## Evidence standard

- Every finding cites file:line(s) and a **measured** count where the claim is about frequency
  (grep command + number).
- Every finding proposes a concrete change and states its **cost** (files touched, pinned/cited
  names affected with counts, rebuild fan-out) and its **payoff in reader terms** ("to change the
  BN backward you currently edit 4 files; after, 1").
- Don't propose renaming pinned/book-cited names unless the payoff is large; a legend or docstring is
  usually the right fix for naming.
- Confidence: **clear** (you checked the consumers and the move is mechanical) · **judgement** (a
  design call the owner should make) · **lead** (worth a look, not fully checked).

## What is NOT a finding

- Formatting, line length, comment density, individual variable names, tactic style (brittleness is
  the other audit's lens), proof length per se.
- "This is complicated" where the mathematics is genuinely complicated and the structure already
  follows it.
- Anything already decided in `planning/` (check before reporting a design call).

## Finding format

```
### B<n>. <short title> — <path>:<line> (+ other sites)

**Problem:** <what a reader trips on, concretely>
**Evidence:** <grep + counts, the sites>
**Proposal:** <the change; for splits, which decls go where>
**Cost:** <files, pinned/cited names, rebuild fan-out>
**Payoff:** <in reader terms>
**Confidence:** clear | judgement | lead
```
