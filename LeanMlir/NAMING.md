# Naming

Declarations in `LeanMlir/` follow Mathlib's
[naming conventions](https://leanprover-community.github.io/contribute/naming.html): theorems
`snake_case`, types and `Prop`s `UpperCamelCase`, other definitions `lowerCamelCase`, and a
`lowerCamelCase` definition keeps that spelling as an atom inside a theorem name. Below are the
suffix meanings that guide leaves implicit, then the local conventions and exemptions.
`scripts/gates/name_lint.py` checks the parts a regex can see.

## Suffixes

| Suffix | Means |
|---|---|
| `_def` | restates a definition (`foo = <its body>`), for rewriting |
| `_apply` | evaluates a function or bundled map at an argument |
| `_iff` | the statement is an `Iff`; a one-direction lemma does not carry it |
| `_eq` | the statement is an equality; an inequality does not carry it |
| `_differentiable` | the statement is `Differentiable`; never abbreviated to `_diff`, which in older names meant a difference (`head_diff`) |
| `_correct` | a VJP witness's backward is the Jacobian-transpose contraction (`reluHasVJP_correct`) |
| `_backward` | a witness's backward, unfolded to a closed form |

The suffix rules follow the TauCeti addendum to the Mathlib guide.

## Local conventions

- **VJP witnesses are data**, because `HasVJP` carries the backward function. A witness is a
  `lowerCamelCase` definition named for its function first and then its rank:
  `reluHasVJP`, `rowSoftmaxHasVJPMat`, `conv2dHasVJP3`, `maxPool2HasVJPAt3`,
  `resnet34ForwardBFullHasVJPAt`. Theorems about a witness keep its name as an atom:
  `reluHasVJP_correct`.
- **Rank converters** live on the structure, in dot form: `HasVJP3.toHasVJP`,
  `HasVJPMat.toHasVJP`, `HasVJPAt3.toHasVJPAt`, `HasVJP.toHasVJPAt`.
- **Composition** of witnesses: `vjpComp`, `vjpCompAt`, `vjpCompDiffAt`, `vjpMatComp`.

## Exemptions

- **Generated certificate tables.** Files whose header says they were generated
  (`scripts/certs/*.py`) name their witnesses by index, e.g. `aczTFe4_0_12` and `hrelTFe4_0_12`.
  Nobody looks these names up by guessing. Renaming them would mean changing each generator's
  templates and re-running it to show its output is still byte-identical. The lint skips these
  files.
- **Paper-letter constants.** A concrete weight or tensor keeps the letter from the text
  (`W1`, `T0`, `X`), and so do theorems about it (`W1_den_certified`). These, and the other names
  left before this convention, are listed in `scripts/gates/name_lint_baseline.txt`, which may
  shrink but never grow.
