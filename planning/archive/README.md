# planning/archive — the lab notebook

Every closed planning doc lives here. Since 2026-09-08 (`planning/cleanup_backlog.md` §1) the
top level of `planning/` holds the standing docs (`cleanup_backlog.md`, `tour_realignment.md`),
the OPEN threads' plans (opened 2026-09-09: `detector_v5_next.md`, `orin_rerun.md`,
`lm_demos_modernization.md`, `brats_25d_3d.md`, `ci_followups.md`, `r34_ablation_seeds.md` — each cites its archived predecessor by full path rather than moving
it) and `mathlib_upstream_drafts/` (the Mathlib upstreaming sources that
`LeanMlir/Proofs/Foundation/UpstreamDraft.lean` mirrors while those PRs are in flight). Everything else — the ImageNet runs, the detector, the diffusion
demos, the proofs-tier campaigns, the float tier, and the closed-thread logs archived before
that date — sits here, flat, under its original name. Nothing here is deleted or wrong; a doc
reads as history because it is history.

## The rule

* A session that reopens a thread pulls its doc back up (`git mv planning/archive/x.md
  planning/`) and repoints every citation in the same commit; when the thread closes it comes
  back the same way.
* Citations spell the full `planning/archive/<doc>.md` path — from Lean docstrings, the
  workflows, `lakefile.lean`, `tests/AuditAxioms.lean`, the scripts, and the docs in here.
  Three emitters bake the path into committed artifacts (`ViTRender.lean` into the four
  `verified_mlir/vit*ema*` renders, `jax/Jax/Codegen.lean` into `jax/generated/`), so a repoint
  of those regenerates the artifact too. Raw `runs/*.log` files keep whatever path the trainer
  printed at the time.
* Section numbering inside a doc is load-bearing (`§3d(b)`, `§4c(c)`, …): docstrings cite
  sections, so an archived doc's numbering is never rewritten.

## Non-md residue

`conv3d_spike.mlir` — the hand-written 3-D conv spike behind the "IREE compiles conv3d" claim
in `brats_demo.md` and `unet3d.md`; the re-run command in those docs points here.
