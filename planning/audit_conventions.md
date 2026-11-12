# House conventions -- lean4-mlir

confirmed_by: repo owner
confirmed_at: 2026-11-12

Drafted by the auditing session from the repo's TRUST.md and module headers, then confirmed item by
item by the repo owner. Treat a delta that matches one of these as CONVENTION, not CONTENT.

This is the conventions file for the formal-companion faithfulness audit
(<https://github.com/mathlib-initiative/formal-companion>, the `cf` tool). After `cf onboard`,
copy it over the placeholder the tool writes at `companions/lean4-mlir/conventions.md` in the
audit workspace. With it in place a run no longer needs `--conventions-unconfirmed`. Update it
here when a convention changes, and re-confirm.

## 1. The book
- The "book" is the repo's own leanblueprint manuscript (`blueprint/src/content.tex`). Item numbers
  like "Theorem 6.4" were assigned by the audit tool and are NOT printed in the text: locate an item
  by its `\label` and the line given with it, never by grepping the number.
- The repo has no `axiom` declarations (checked: none in any `.lean` file; `tests/AuditAxioms.lean`
  holds every audited declaration to `propext`, `Classical.choice`, `Quot.sound`). An
  `axiom`-styled or `ax:`-labelled item is a definition or a proved fact in Lean.
- One blueprint item may list several Lean declarations: per-layer or batched variants of one
  statement, and concrete instances ("Witness: ...") of a general one.

## 2. Standing conventions
- Everything is over the reals. Floats enter only through an explicit `FloatModel` hypothesis; every
  float theorem is "for all M".
- Smooth-input, positivity and magnitude-window conditions are explicit hypotheses the caller
  supplies. A theorem that carries them is not weakened by them when the prose states or cites them.
- A strictly more general Lean statement that specialises to the prose's is not a defect.
- Training masks (stochastic depth, classifier dropout) are optional binders on the step ties of
  every net whose renders carry them (ResNet-50, MobileNetV2, MobileNetV4, EfficientNet, ViT,
  ConvNeXt): `none` is the drop-free statement, `some` the masked renders'. The mask VALUES are
  binders: how the driver samples them is not modelled.
- Inference forwards (BatchNorm at frozen running statistics) are stated without hypotheses, per
  example for the ResNets and MobileNetV2 and at batch N for MobileNetV4 and EfficientNet. Their
  weight records carry conv-bias fields the artifacts bind to zero.
- Optimizer tails (the update rule, EMA, gradient accumulation, the clip) are separate certified
  statements (`thm:optimizer_tails_faithful`, `thm:lamb_clip_faithful`, `thm:accum_mean_step`,
  `thm:rmsprop_faithful`, `thm:ema_faithful`), not part of the gradient-node ties. No single Lean
  theorem composes a net's tie with its tail; prose that claims one IS a content finding.
- The project does not rederive its foundations: Mathlib, the Lean kernel and the StableHLO
  denotation `den` are taken as given.

## 3. Sorry policy
- Proofs contain no `sorry` (checked: no `sorry`, `admit` or `native_decide` token outside comments).

## 4. Trained-weight witnesses
- "Witness" sentences refer to narrow trained networks of the chapter's architecture on a real
  MNIST image; their smallness (width, step size) is deliberate, since kernel-checked arithmetic
  does not scale to real-world network sizes. A hypothesis that holds only at such a witness is
  satisfiable, hence not vacuous.
