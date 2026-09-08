# Cleanup backlog — what is left after the float second pass, and the planning-folder sweep

**Standing doc, opened 2026-09-08 after `41d52b2` landed on `main` (the 13-commit
`cleanup/2026-09-08-unify-and-float-chop` branch: bf16 fold nodes, one opaque-prefix chain, the
float tier cut to its model core, −19.9k lines). Every item below is a separate session and a
separate commit; none blocks another. Do §1 first — it is the one that changes where the other
items' references point.**

Gates for every item, unchanged from `float_second_pass.md` §5: `lake build Certs`,
`lake env lean tests/AuditAxioms.lean` (3-axiom clean), `lake exe docstring-checkrefs`,
`python3 scripts/check_audit_coverage.py`, `python3 scripts/check_render_coverage.py`,
`git diff verified_mlir/` empty. Prose-only items still run the docstring gate (it resolves
every backticked identifier against the environment) and `lake exe blueprint-checkdecls` if the
book is touched. Land by fast-forward, never a merge commit.

## 1. The planning folder: move everything to `archive/`, pull back on demand

`planning/` holds 142 standing docs, 61.7k lines; `planning/archive/` holds 18, 8.2k. Almost all
of the 142 are finished threads (the ImageNet runs, the detector, the diffusion demos, the
proofs-tier campaigns, the float tier) that read as current because they sit at the top level.
The decision (2026-09-08): move every `planning/*.md` except this file into `planning/archive/`,
and pull a doc back up only when a session reopens its thread.

Mechanics, one commit:

* `git mv planning/*.md planning/archive/` (this file excepted). Keep the archive flat; the
  existing 18 files already are.
* **490 lines in `LeanMlir/` and `tests/` cite a `planning/<doc>.md` path** in docstrings, and
  `CLAUDE.md` / `README.md` / `formalization.yaml` / `blueprint/src/content.tex` a handful more.
  `docstring-checkrefs` skips file names, so nothing goes red — but the paths become lies. One
  `sed 's|planning/\([A-Za-z0-9_-]*\.md\)|planning/archive/\1|g'` over those trees (excluding
  paths that already say `planning/archive/`) keeps every citation truthful. Run the docstring
  gate after; it is cheap.
* The memory index outside the repo (`~/.claude/.../memory/`) names planning paths too; those are
  background notes, update on next touch.
* The most-cited docs, i.e. the ones a future session is likeliest to want back at the top level
  (citing files, from Lean/tests/book/yaml): `mnv4_verified.md` 30, `xla_pjrt_handoff.md` 27,
  `proofs_tier_to_paper_nets.md` 23, `stochastic_depth.md` 16, `ema.md` 14,
  `rsb_a3_r50_verified.md` 11, `grad_clip.md` 11, `renderer_convergence.md` 10,
  `bf16_renderer.md` 10, `yolo_fpn.md` 9, `next_session_pipeline_then_r50.md` 9,
  `float_second_pass.md` 8. Forty-nine docs are cited by nothing outside `planning/` and move with
  zero fallout.
* Non-`.md` residue at the top level: `conv3d_spike.mlir` and `mathlib_upstream_drafts/` —
  decide each (the drafts are a thread of their own; the spike is an artifact of a closed one).

**DONE 2026-09-08.** 142 docs and `conv3d_spike.mlir` moved; `mathlib_upstream_drafts/` stays
(its thread, `UpstreamDraft.lean` and the in-flight PRs, is open). The citation surface was 535
files, not the 490 lines above: `apps/`, `demos/`, `ffi/`, `jax/` (108 lines of it in
`jax/generated/`), `scripts/`, the five workflows, the `runs/` READMEs, and 292 cross-references
inside the docs themselves. Three emitters bake the path into committed artifacts (`ViTRender.lean`
into four `verified_mlir/vit*ema*` renders, `jax/Jax/Codegen.lean` into `jax/generated/`), so
source and artifact moved together and both drift guards were re-run. Four line-wrapped cites and
two bare `archive/x.md` links were fixed by hand; six raw `runs/*.log` files keep the old path.
`yolo_demo_v3.md`'s five cites were already dead (deleted at `a0a33a3`, folded into
`yolo_final.md`) and now say so. The blueprint cites no planning path, and there is no
`CLAUDE.md`. The memory index still names `planning/<doc>.md` in 37 notes, per the rule above.

## 2. `tests/AuditAxioms.lean` as a log

4997 lines: 1760 `#print axioms` and 2937 comment lines, and the comments are the debugging
story ("this comment used to say…", per-session narratives, numbers that were later deleted).
The rule is that process goes in `planning/`; the audit should be prints under one-line section
headers. Archive the narrative into `planning/archive/audit_axioms_log.md` (it is genuinely the
best log of what each tier found), cut the file to roughly a third, same prints. Zero proof risk;
the only gate that reads it is the three-axiom run. A few historical paragraphs there still name
files deleted on 2026-09-08 (`EnetFloatBridge`, `BnEvalRuntimeFloatBridge`, `SEBackFloatBridge`,
`ViTAttentionFloatBridge`, `FloatBudgetEnvMBConv`); they go with the narrative.

**DONE 2026-09-08.** 4997 → 2426 lines: the 1760 prints and 191 imports byte-identical and in
the same order; each of the 362 comment blocks is one header line (34 rule lines, 17 banner
paragraphs and 312 narratives cut; 33 were one line already), plus a three-line pointer under
`open Proofs`. The narrative is `planning/archive/audit_axioms_log.md` (4641 lines): the same 362
headers in file order, each block's prose verbatim, then the theorems it heads. Two identical
"Increment 1 keystone" labels are now told apart by subject. Gates: 1760/1760 verdicts on the three
axioms, CI's print count, audit coverage, docstring gate.

## 3. The float prose left in the kept tier

21 files under `Foundation/`, `Architectures/`, `Codegen/`, `Certificates/`, `Training/` still
describe their subject as "the float bridge" or cite `floatBridges_*` names in prose. The
docstring gate is green because those identifiers resolve (the model core still defines some), but
the framing is stale: the chains the ties are about are the ℝ leaves now
(`Foundation/BackwardMaps.lean`, `Architectures/ChannelLNBack.lean`, `Foundation/*BackChains.lean`).
Prose-only, one commit; grep `-il 'float bridge\|FloatBridges\b\|floatBridges_'` to list them. The
tie headers that begin "The A3 backward float bridge `X` (…FloatBridge.lean) proves…" are the
worst offenders; three were rewritten during the second pass and read as the template.

**DONE 2026-09-08.** 64 passages in 14 files (the six §B tie files, `EvenKernelConvBack`,
`ConvNeXtWholeBackCertifiedTie`, `ConvNeXtFullT`, `MaxPool3s2`, `DropPath`,
`EfficientNetRenderPCEval`, the two `SgdDescent*`): every "float-bridge `X`" is "the backward
map `X`" or "the chain", every "deployed-float ≈ transcription" is "the chain IS the certified
VJP", and eleven cites of deleted objects (`floatBridges_mbconvBody`, `floatBridges_convBack`,
`floatBridges_mhsaBack`, `floatBridges_vitBlockBackPR`, `floatBridges_chanLNTensor3Back`,
`floatClose_bnBack`, `r34_floatBridges`, `r34Forward`, `resnet34Forward_full_pc_eq_skeleton`,
`floatBridgesTo_convNextStageChK`, `FloatBridgesTo.ofEq`) now name what exists. Two stale
non-float claims in the same paragraphs went with them (ResNet-34 has its shape check,
`resnet34Forward_full_pc_eq_chain`). Kept: the ℝ leaves' one-line provenance notes, the
Training/Certificates prose about `FloatBridge.lean` (live), `MlpCanonical`'s existential
statement (§8). ⚠ The premise above was wrong: the gate was green because it never LOOKED —
`docstring-checkrefs` resolves only names carrying a `projectMarkers` substring, and no float
name did. The markers now include `floatBridges_` / `floatClose_` / `FloatBridges` /
`FloatClose`; that admitted 96 more citations and three more dead ones in `Float/`, fixed here.

## 4. `formalization.yaml`

Deferred by user decision during the second pass. Two things: 4c is a stub whose only content is
that it was deleted, and 4d carries a "what was deleted, and why" paragraph plus dated deletion
notes in the style step 1 set. The book's rule applies (the sentence describes what exists; the
past is not mentioned); the yaml is an index and should read the same way. ⚠ The §1–6 / 4b / 4c /
4d numbering is load-bearing for cross-references in docstrings and the book — renumber nothing,
empty 4c to one line if a section must remain.

**DONE 2026-09-08.** 4c is one line, a present-tense disclosure (no theorem bounds a whole-net
float error against a margin); 4d's dated title, its "what was deleted" paragraph and the twin
note are gone, and its model paragraph names the two closeness forms the surviving tier is
stated on (`FloatClose` for the `floatClose_*` lemmas, the r34 stage fold and the bf16-mixed
bounds; `FloatBridgesTo` for the per-op instances, the combinators and the CIFAR chains -- the
old text put the bf16 bounds and the descent chain on `FloatBridgesTo`, which nothing is). The
status block's "whole-net float bridges carrying window and modulus" (none exist) and the
alignment note's deletion sentence are rewritten the same way. Numbering untouched: 4c is cited
only by the yaml's own alignment note, 4d by `FloatComposeBridge.lean` twice. 399 → 391 lines,
parses. Left as they were: the dated closure notes in 4e-4g and the padding paragraph, which are
records of what is, not of what was deleted.

## 5. Two small Lean tidies

* `perRowIdxFlat` and `perRowFlatPR` (`Foundation/BackwardMaps.lean`) are the same map,
  definitionally; two names survive because the ties spell both. Pick one (`perRowFlatPR` is the
  documented one), rename the other's uses in the ConvNeXt/ViT/BN ties, delete it.
* `Codegen/BnInputBridge.lean` and `Codegen/Resnet34BlockBridge.lean` are float-closeness bridges
  whose only consumer is `Float/FloatComposeBridge.lean` (which feeds the bf16-mixed compose
  bridge). Move them to `Float/`. Rename-only; the lakefile roots and the audit imports follow.

**DONE 2026-09-08.** `perRowIdxFlat` and its `_apply` lemma are gone; `perRowFlatPR`'s docstring
carries the block-diagonal reading, and the four sites (`rowLNVecFlatBack`'s body, one `unfold`
in `ConvNeXtBackB0`, two docstrings) spell `perRowFlatPR`. `BnInputBridge.lean` and
`Resnet34BlockBridge.lean` are `Float/` modules (15 files there now); their two imports, the
two lakefile roots and the two audit imports followed. Nothing else named either module.

## 6. `lakefile.lean`

4040 lines, 1356 of them comment essays attached to individual roots. A root list should be a
list; the essays belong in the modules' own headers (most already duplicate them). Cosmetic, but it
is the file every session touches, and `check_audit_coverage.py` parses its root lists with a
regex that the essays' brackets already defeated once (the script strips `--` comments for that
reason).

**DONE 2026-09-08.** The `Proofs`, `Certs` and `CertsHeavy` root arrays are bare lists, one root
per line (4041 → 3036 lines); every array is the same ordered list it was. The 187 per-root
comments (1013 lines, all on modules that carry a `/-!` header of their own) are
`planning/archive/lakefile_roots_log.md`, verbatim, under the module each preceded; the two lib
docstrings point there, and the `Certs` one says 201 roots reaching 235 modules (~153k lines)
instead of "155 files, ~87k lines". The 343 `--` lines outside the arrays (exe notes, section
banners, the doc-gen4 note) are not root essays and stay.

## 7. The one real gap: a batched ViT T6

`vitInputGradK` (`Foundation/ViTBackChains.lean`) is per-example; its `N` is the token count, not
a batch. Every other net has a batched whole-net certified backward tie (`*InputGradB_eq_*_vjp`).
ViT has no BatchNorm, so the batched chain is the `StableHLO.batchMap N` lift of what exists and
the tie should close the way ResNet-34's batched pool did — field by field, `rfl` at the leaves.
The batched T3 tie (`ViTTiePoCGB`) already spells the batched forward this reverses. Not cleanup,
but it is the last "every net, every tier" claim the yaml cannot yet make.

**DONE 2026-09-08.** `vitInputGradKB` (`ViTBackChains.lean`): the five stages of `vitInputGradK`
each lifted over a variable batch `B` — `batchMap B` for the input-independent head, CLS-scatter
and patch-embed leaves, `batchMapAux B` for the tower and final-LN backwards at the batched saved
activations, saved stage by stage (`batchMap B (f ∘ g)` and `batchMap B f ∘ batchMap B g` agree
only up to `batchMap_comp`, not `rfl`). `ViTWholeBackCertifiedTieB.lean`: four leaf ties (the
patch-embed one `rfl`, the others one rewrite of the per-example tie at one example's row), a
`vjp_comp_diff_at` apex over `batchMap_has_vjp_at` witnesses, the tie, the shape check
`vitForwardKVB_eq_chain`, the transfer to the committed `batchMap_has_vjp (vitForwardKV …)`
through `HasVJPAt.backward_unique_of_eq`, the `∑ pdiv` reading, and the ViT-Tiny capstone with
`B` a binder. Thirteen declarations, all on the three axioms; no smoothness hypothesis, only
`0 < ε`. ⚠ The claim above was off by one: ConvNeXt has no batched T6 either (its ImageNet
artifacts never had a batched fold; `renderer_convergence.md` leg 3). LayerNorm is per-example,
so it would close the same way, and it is the last `*InputGradB` gap.

## 8. Lower priority

* `Float/FloatComposeBridge.lean` (776 lines) still carries `FloatBridges` / `FloatBridgesTo`
  scaffolding beyond the `FloatClose` chain `ConvMixedComposeBridge` composes
  (`floatClose_r34_stages`, `floatClose_bn`, `floatClose_residualBlock`, …). What
  `DepthwiseFloatBridge` and `FloatSubnormalBridge` use is the `.comp` / `.residual` combinators;
  the rest can go. Measure with the same script the second pass used (declaration names of the
  file grepped against every other kept file).
* `Foundation/MlpCanonical.lean`'s backward float statement is `mlpInputGrad_floatBridges`, on the
  existential `FloatBridges` predicate the yaml itself says "constrains nothing — do not cite as
  budgets". Either restate it on `FloatBridgesTo` or drop it, and `Float/LinBackFloatBridge.lean`
  goes with it (its only kept consumer).
* Probe scripts under `scripts/` that mention float names (`transcendental_probe.py`,
  `kernel_faithfulness_probe.py`, `margin_probe.py`, …) point at `FloatBridge.lean` and the fp8
  files, which stay. Leave them.

## 10. The `Proofs/` layout — the buckets hold, the per-net scatter does not

Asked 2026-09-08: touch the layout again, or does it make sense? Measured:

| dir | files | lines | what it is |
|---|---|---|---|
| `Foundation/` | 71 | 29.4k | 26 generic (Tensor, IR, VJP calculus, CertifiedChain, OpaquePrefix, BackwardMaps, BatchMapVJPAt, DataParallel, …) + **45 net-specific** (ResNet34/50 ties, MobileNet ties, B0 ties, the `*BackChains`, `*TiePoCB`, `*FaithfulPoCB`) |
| `Architectures/` | 77 | 41.7k | 10 generic ops (CNN, BatchNorm, LayerNorm, Attention, Depthwise, SE, Residual, MaxPool3s2, ChannelLNBack, DepthwiseBackCertifiedTie) + **67 net-specific** |
| `Codegen/` | 31 | 32.8k | StableHLO AST/denotation, the renderers, the optimizer steps — coherent |
| `Certificates/` | 35 | 41.4k | Lipschitz / smoothing / IBP — coherent |
| `Training/` | 15 | 20.7k | descent theorems and seals — coherent |
| `Float/` | 13 | 5.6k | the rounding model and its results — coherent after the second pass |

Verdict: the six buckets are right, and four of them are clean. The problem is that **per-net
proof files are split between `Foundation/` and `Architectures/` by accident of when they were
written, not by kind** — 11 `*CertifiedTie*` files in one, 6 in the other; ResNet-34's forward is
`Foundation/ResNet34.lean`, its tie is `Foundation/Resnet34BackCertifiedTie.lean`, MobileNetV2's
block tie is `Architectures/MobileNetV2BackCertifiedTie.lean` and its whole-net tie is
`Foundation/MobileNetV2WholeBackCertifiedTie.lean`. Two naming layers on top: `Resnet*` (6 files)
beside `ResNet*` (25), and 36 files named `*PoC*` for what is now the production tier
(`FaithfulPoC`, `TiePoC`, `TiePoCGB`, `TiePoCB`, `FaithfulPoCG`, `FaithfulPoCGB`).

The fix that captures nearly all the value without re-inventing the buckets:

1. **One home per net.** Move the 45 net-specific `Foundation/` files into `Architectures/`
   (or, cleaner, a new `Nets/<Net>/` tree with one directory per family — ResNet, MobileNet,
   EfficientNet, ConvNeXt, ViT, the CIFAR/MNIST small nets), leaving `Foundation/` = generic
   infrastructure and `Architectures/` = generic ops. The `*BackChains` leaves written on
   2026-09-08 went to `Foundation/` only because the whole-net ties were there.
2. **Normalize `Resnet` → `ResNet`.** Six files.
3. **Drop `PoC` from production names**, with a suffix that says what the file is (`Faithful`,
   `Tie`, `TieB`, `TieGB`). 36 files; every tie docstring and the book cite these names.

All three are `git mv` + citation updates — no proof changes — but the citation surface is the
whole point of doing it as ONE session with the same `sed` machinery as §1: 490 `planning/` path
lines in Lean (§1), the lakefile roots, the audit imports, the yaml's 23 `file:` fields, the
book's 15 `\texttt{*.lean}` mentions, `proofs.yml`'s 24 `lake env lean LeanMlir/Proofs/…` drift
lines, `regen_verified_mlir.sh`'s writer list, `check_render_coverage.py`, the memory index. Do it
after §1–§3 (they rewrite the same prose) and before §7. ⚠ The renderers' module names are
baked into `verified_mlir/` provenance comments? — check `scripts/check_render_coverage.py` and the
drift guard before moving anything under `Codegen/`; the safe version of this item leaves
`Codegen/` alone.

## 9. Not on the list, deliberately

The 15 `Float/` files (13 plus the two §5 moved in) are the model core (the rounding model, `Binary32Instance`, the
subnormal bridge, the bf16-mixed and fp8 results) plus the ResNet-34 forward chain the bf16-mixed
compose bridge builds on. The book's "Finite precision" section argues from exactly these. The
saturation constants for GELU and Swish (`geluScalar_lipschitz`, `swishScalarDeriv_abs_le`) were
deleted with the budgets; no smoothing or Lipschitz certificate ever used them, and they are one
`git show 55b0630:LeanMlir/Proofs/Architectures/GeluSaturation.lean` away if a GELU Lipschitz
certificate is ever wanted.
