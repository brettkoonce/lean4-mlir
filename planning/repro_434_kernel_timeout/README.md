# Repro: `(kernel) deterministic timeout` on Lean 4.34.0, absent on 4.32.2

## What
`LeanMlir/Proofs/Nets/ConvNeXt/ConvNeXtWholeBackCertifiedTieB.lean` builds on Lean **4.32.2** and
fails on **4.34.0** with `(kernel) deterministic timeout` on `convNextForwardTChB_has_vjp_at`
(the twelve-stage batched VJP apex, eleven nested `vjp_comp_diff_at`s).

The three `unknown constant 'Proofs.convNextForwardTChB_has_vjp_at'` errors that follow are
**cascade** — the decl never entered the environment — not independent failures.

## Bisection (truncate the apex to k composition levels above the stem pair)

| levels | Lean 4.32.2 | Lean 4.34.0 |
|---|---|---|
| 2 | builds, 4s | builds, 2s |
| **3** | **builds, 2s** | **kernel timeout, ~162s** |
| 4 | builds, 2s | — |

A cliff, not a gradient: one extra composition level goes from 2s to non-terminating, and 4.32.2
handles the same level in 2s. So it is not the particular stage (`convNextStageChK 3 w.s2`) — that
same stage is instant on 4.32.2.

## Not the cause (ruled out by measurement)
* **Raising `maxHeartbeats`** — at 4000000 the full module ran 37+ min, climbing monotonically to
  86.8 GB RSS, no result (killed to protect a training run). Versus ~18 min and modest memory on
  4.32.2. So it is a blow-up in kernel reduction work, not a too-small budget.
* **Nesting depth alone** — a Mathlib-free skeleton (dependent `V n := Fin n → Nat`, a structure
  with a function + proof field, a composition combinator, symbolic `B * 96` indices) compiles
  instantly on both toolchains at depth 5, 11 and **16**. So depth + symbolic batch indices are not
  sufficient; the Mathlib half (`DifferentiableAt ℝ`, `pdiv = fderiv ℝ`, the `∑ j : Fin n` in
  `HasVJPAt.correct`) is load-bearing.

## Reproducing
`generate_truncation.py K OUT` writes a truncated copy of the module (its prefix through the
witness definitions, then a K-level apex). Compile with the project's `LEAN_PATH`:

    lake env printenv LEAN_PATH > lp.txt
    python3 generate_truncation.py 3 t3.lean
    LEAN_PATH="$(cat lp.txt)" lean t3.lean

Requires the module's import cone (`ConvNeXtWholeBackCertifiedTie`, `BatchMapVJPAt`) built, so it
is not Mathlib-free — the Mathlib content is part of the trigger.

## Upstream note
Lean core rc2 -> 4.34.0 final is only 3 backport commits (InfoTree/editor + mimalloc), so this is
present throughout 4.34, and the separate `rw`-pattern regression from this repo still reproduces
on 4.35.0-rc1.
