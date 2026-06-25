import LeanMlir.Proofs.Binary32Instance

/-!
# Trusted ℝ→Float32 bridge: the NAMED-axiom footprint audit

`tests/AuditAxioms.lean` proves the whole VJP/render/tie suite is closed under the
Lean-core triple `[propext, Classical.choice, Quot.sound]` — **zero project axioms**.

`LeanMlir/Proofs/Binary32Instance.lean` is deliberately different: it names the IEEE-754
standard-model rounding assumption as two explicit axioms (`ieeeRnd`, `ieeeRnd_err`) — the
single thing Lean cannot prove about opaque hardware `Float` — and uses them to discharge
concrete binary32/fp8 corollaries (argmax preservation, one float-SGD descent step). It is
kept OUT of the zero-axiom suite (NOT imported by `Proofs`/`AuditAxioms`).

This file is its audit. The CI step that runs it (`lake env lean tests/AuditTrustedBridge.lean`)
asserts each capstone depends on **exactly** the 3-axiom triple PLUS those two named trust
axioms — and NOTHING else (in particular, no `sorryAx`). That turns "we added an axiom" into
a CI-enforced, *bounded* trust surface: the bridge can lean on `ieeeRnd`/`ieeeRnd_err` and
nothing further.
-/

-- gap 2: fp8 argmax preservation, unconditional on the named binary32/fp8 models
#print axioms Proofs.binary32_e4m3_argmax_preserved
#print axioms Proofs.binary32_e4m3_argmax_small
-- gap 3: one binary32 SGD step provably decreases the real loss (smallness conds discharged)
#print axioms Proofs.binary32_linear_sgd_descends_concrete
