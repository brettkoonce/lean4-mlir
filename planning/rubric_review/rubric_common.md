# Rubric review 2026-09-30 — shared protocol (read first)

You are one of eight auditors, each on one slice of this Lean 4 repository, running the TauCeti
review rubrics (`planning/rubric_review/rubrics/`, vendored from
`TauCetiProject/TauCetiReview@dcc918ab`) as a **whole-tree audit**, not a PR review. Read
`rubrics/_common.md` and every angle file (`correctness`, `reuse`, `scope`, `attribution`,
`api-design`, `generality`, `placement`, `naming`, `documentation`, `proof-quality`, plus
`references/naming-conventions.md`). Their standards apply; this file says how they translate here.
**Report findings only. Edit nothing in any checkout.**

## Checkouts — two, and one is not yours

- **Audit tree**: `/home/skoonce/lean/klawd_max_power/lean4-jax-mlir-rubric-review` (branch
  `rubric-review` at `55ad3a5a`). Read sources here. Cite paths relative to it.
- **Main checkout**: `/home/skoonce/lean/klawd_max_power/lean4-jax-mlir`. ⛔ Another agent is working
  there. Use it ONLY read-only for (a) Mathlib ground truth at
  `.lake/packages/mathlib/Mathlib` (Batteries at `.lake/packages/batteries`, core at
  `~/.elan/toolchains/leanprover--lean4---v4.34.0/src/lean/`), and (b) typechecking a scratch file
  with `cd <main> && timeout 600 nice lake env lean <your scratch dir>/x.lean` against its built
  oleans. **Never** `lake build`, `lake clean`, `lake update`, `git` anything, or write a file there.
  Typecheck only to confirm a correctness or reuse claim you would otherwise leave as "suspected";
  at most one `lake env lean` of yours at a time.
- Scratch files: your own directory under
  `/tmp/claude-1000/-home-skoonce-lean-klawd-max-power-lean4-jax-mlir/a7bffe59-a150-4b04-97d5-40135dabae75/scratchpad/<slice>/`.
- ⚠ The local `grep` may honour .gitignore; for `.lake/packages/mathlib` use `grep -r` on explicit
  paths, and if a search looks suspiciously empty confirm with `find … -exec grep`.

## How TauCeti's angles translate to this repo

This is not a Mathlib-downstream maths library; it is a verified ML compiler + training stack whose
proof tier ties rendered StableHLO/MLIR to real-analysis specs (VJPs, folds, step ties, seals,
float budgets, certificates). Translate as follows:

- **"The PR" = your slice as it stands at HEAD.** Line `0` = file-wide.
- **Roadmap (scope angle)** = what the repo claims to deliver: `formalization.yaml`, the book
  (`blueprint/src/content.tex`), `TRUST.md`, `README.md`, `LeanMlir/Proofs/README.md`, the lakefile
  lib roots (`Certs`, `CertsHeavy`, `Proofs`, …), `tests/AuditAxioms.lean` pins, and the comparator
  tier (`scripts/gates/gen_comparator_tier.py`). A declaration is "on the roadmap" if a path of
  imports/uses reaches one of those. Report material on no such path (dead, or a satellite whose
  target never got closer), and files doing more than one job. Do not re-report the dead-code sweeps
  already landed (see history).
- **Correctness** = the rubric verbatim, plus this repo's specific vacuity shapes: `∃`-modulus
  statements satisfiable by a trivial witness (the old `FloatBridgesTo ∃L`), `HasVJP`-style
  structures whose `.correct` is satisfied by a canonical witness so "`_correct`" says nothing about
  the rendered code, a tie that holds at a width/class-count other than the shipped artifact's, a
  predicate (e.g. a smoothness/distinctness hypothesis) that fails on real data so the capstone never
  applies, and content moved into hypotheses (a bound assumed rather than proved).
- **Reuse** = Mathlib/Batteries/core AND the repo's own shared kits (`Foundation/`, `Architectures/`,
  `RenderKit`, `CertLayer`, `CertifiedChain`, `GradNodesB`, `BatchedStages`, `DataParallelSyncKit`,
  `SgdNodes`, `ParamGrad`, `ParamGradNodes`, …). New code that bypasses a kit is a finding.
- **Attribution** = papers the nets/recipes/proofs follow (ResNet, MobileNetV2/V4, EfficientNet,
  ConvNeXt, ViT/DeiT, RSB/timm recipes, LAMB, Chan variance, Clopper–Pearson, Cohen et al. randomized
  smoothing, CROWN/IBP, Silver et al., …), vendored or adapted Mathlib material
  (`planning/mathlib_upstream_drafts/`), timm/JAX code a Lean spec mirrors. Credit belongs in the
  module docstring. Do not demand credit for routine work.
- **API design / generality / placement / naming / documentation / proof-quality** = the rubrics
  verbatim. Naming conventions specific to this repo: `LeanMlir/NAMING.md`,
  `scripts/gates/name_lint.py`, and memory "VJP defs thing-first (`reluHasVJP`)", "per-net files named
  `*Fold*` / `*StepTie*`". Consistency with adjacent declarations beats the Mathlib ideal.
- **Compatibility policy** holds verbatim: never propose an alias/shim; a rename moves every consumer.

## What CI already enforces (don't re-report)

Build of every lib, `tests/AuditAxioms.lean` axiom pins, the comparator tiers,
`docstring-checkrefs` (backticked names resolve), `name_lint.py`, `check_target_names.sh`,
`import_audit.py`, `blueprint_uses.py --check`, `regen_verified_mlir.sh check` (artifact byte
identity), book xrefs. A missing mechanical check is a "gap for the humans" line, not a finding.

## History you must not repeat

The tree has had, in order: Mathlib-reuse v1 (`planning/archive/mathlib_reuse_audit.md`,
~24.7k lines out), audit census, proof-quality cleanup (`planning/proof_cleanup.md`,
`planning/proof_cleanup_audits/`), reuse v2 + clarity (`planning/audit_v2.md`,
`planning/audit_v2_rubrics/slice_notes.md`), correctness audit 2026-09-24 (report summarised in
`planning/doc_honesty_pass.md` + `planning/doc_audit/`), naming pass 09-25, placement cleanup
(`planning/placement_imports_cleanup.md`), API-design audit 09-26 (`planning/api_design_audit.md` —
this WAS the api-design rubric; its §0 rules and "checked and rejected" notes bind you), doc
honesty pass (`planning/doc_honesty_pass.md`), `planning/cleanup_backlog.md`.

- Skim the section(s) of those docs for your slice BEFORE auditing. Do not re-report a landed or
  rejected item unless the code changed and the rejection reason no longer holds — then say what
  changed. An item those docs list as OPEN and still present may be restated in one line with its
  doc reference (so the plan can farm it), marked `(carried: <doc> §x)`.
- Code landed after 2026-09-26 (`git diff --stat 373059db HEAD`) has had NO audit from any angle;
  weight it heavily.
- ⛔ Known traps (don't propose fixes that hit them): `simp only [<rfl lemmas>]` / `den` in a simp set
  makes the kernel unfold everything; literal-width `rfl` or `fun_prop` on block lemmas at numeral
  widths → deep recursion / timeouts; IR-spelled backwards (`reindexHasVJP`, `broadcastFlatHasVJP`)
  are matched by `rfl` in graph ties and cannot become aliases; root files (`Foundation/Tensor.lean`
  ~423 downstream modules) are expensive to touch; `@[irreducible]` is deliberate where present.
- **Generated files**: skip any file whose header says `Generated by`, `AUTO-GENERATED`,
  `DO NOT EDIT`, or `@generated`; report the generator under `scripts/` instead.
  `tests/comparator/Challenge*.lean` use `sorry` on purpose.

## Pinned names are expensive

Declarations are cited by `tests/AuditAxioms.lean`, `blueprint/src/content.tex`,
`formalization.yaml`, `.github/workflows/*.yml`, `scripts/gates/gen_comparator_tier.py`, and
docstrings. For any rename/move/delete, `grep -c` those and state the cost in the finding.

## Mechanical census (precomputed; use it, verify before citing)

`<scratchpad>/census/census.txt` (scratchpad root =
`/tmp/claude-1000/-home-skoonce-lean-klawd-max-power-lean4-jax-mlir/a7bffe59-a150-4b04-97d5-40135dabae75/scratchpad`):
theorem/lemma spans > 50 lines (157), `change`/`show` with no comment in the 2 lines above (420),
`set_option maxHeartbeats` (53), public `def`/`structure`/`abbrev` with no docstring directly above
(1,723 — heuristic, many false positives), files lacking `/-!`. The census is a line heuristic —
confirm each item you use.

## Output

Write your report to `<audit tree>/planning/rubric_review/slice_<ID>.md`, then reply with a ≤ 15-line
summary: finding counts per angle, the top 5 findings overall, and the report path. Report shape:

```
# Slice <ID> — <name>: rubric review 2026-09-30

## Verdicts
| angle | verdict (approve / request_changes / block) | findings |
(all ten angles, one row each)

## Findings
### <angle>
- **<ID>-<angle-abbrev>-<n>** `path:line` — problem. **Fix:** concrete change. **Evidence:** grep hit /
  line / reasoning. **Cost:** pins/consumers touched, est. lines ±, risk. **Size:** S (<1 h) / M (half
  day) / L (multi-day). (carried: … if applicable)
(block-capable angles first: correctness, reuse, scope, attribution; then the rest in rubric order)

## Checked, not findings
<one line each — what you looked at and why it's clean; saves the next auditor time>

## Gaps for the humans
<missing mechanical checks, rubric mismatches for this repo>
```

Every finding must be independently actionable by a future agent that has not read your report's
other items: file, line, declaration name, the concrete fix, and how to gate it. A short list of
verified findings beats a long list of maybes; "when unsure whether a point clears the materiality
bar, omit it". A clean angle is a useful result — say so plainly.
