import LeanMlir.Proofs.LipschitzCertScorecardFull
import LeanMlir.Proofs.LipschitzCertScorecardSDPFull
import LeanMlir.Proofs.LipschitzCertScorecardSDPFullUncon
import LeanMlir.Proofs.LipschitzCertScorecardIBP
import LeanMlir.Proofs.LipschitzCertScorecardIBPUncon

/-! # Axiom audit — the HEAVY generated certificate corpus (`CertsHeavy`)

The full-input (784-dim) scorecard instances: L2 Tsuzuku + per-pair LipSDP +
IBP L∞, ~90k generated lines of weight/image data and per-image theorems.
Split out of `tests/AuditAxioms.lean` 2026-07-12 together with the
`Certs`→`CertsHeavy` lakefile split: the long-running data-heavy corpus gets
its own workflow (.github/workflows/certs-heavy.yml) so it cannot take
certs.yml/blueprint.yml down. Same gate: every line below must close under
exactly `[propext, Classical.choice, Quot.sound]`. The hand-written engine
cores (ListDot.lean, IntervalBound.lean) stay in `Certs` and are audited by
the MAIN audit. -/

-- FULL-INPUT scorecard (2026-07 audit gap #3, LipschitzCertScorecardFull*.lean): the
-- pooled 49-dim reduction lifted to the genuine 784-dim input (exact k/255 pixels),
-- per-image certificates at pixel-L2 ε = 1/10 AND 3/10 on two 784→16→10 nets —
-- capped σ≤2: 92/100 @0.1 (PGD bracket 93 — within ONE image of the attack bound) +
-- 72/100 @0.3; unconstrained: 76/100 @0.1 → 2/100 @0.3 (the σ-projection is what
-- survives the bigger radius). Engine: ListDot.lean — every 784-term dot is one
-- kernel `dotZ` evaluation (`decide +kernel`, GMP, propext-only; NOT native_decide)
-- transported to the `Fin 784` sums by the once-proved `sum_getD_div` bridge; the
-- pooled recipe's simp sum walk is quadratic in input dim and priced out at 784.
-- Spot-check: the bridge core, both nets' Schatten-8 chains, a Gram entry +
-- wrapper per net, first/middle/last per-image certs at both radii, and the
-- mechanized aggregates. (The raw `gz*` kernel dot facts are propext-ONLY —
-- stricter than the triple — so they'd trip the exact-triple CI grep; they're
-- covered transitively by every entry lemma, e.g. gSF_0_15 below.)
-- (core, audited in AuditAxioms) Proofs.dotZ_comm
-- (core, audited in AuditAxioms) Proofs.sum_getD_mul
-- (core, audited in AuditAxioms) Proofs.sum_getD_div
#print axioms Proofs.LipschitzCertDemo.gSF_0_15
#print axioms Proofs.LipschitzCertDemo.G1SF_eq
#print axioms Proofs.LipschitzCertDemo.H1SF_eq
#print axioms Proofs.LipschitzCertDemo.W1SF_lip
#print axioms Proofs.LipschitzCertDemo.W2SF_lip
#print axioms Proofs.LipschitzCertDemo.mlpSF_lip
#print axioms Proofs.LipschitzCertDemo.G1TF_eq
#print axioms Proofs.LipschitzCertDemo.W1TF_lip
#print axioms Proofs.LipschitzCertDemo.mlpTF_lip
#print axioms Proofs.LipschitzCertDemo.hpreSF0_eval
#print axioms Proofs.LipschitzCertDemo.marginSF0
#print axioms Proofs.LipschitzCertDemo.certSF10_0
#print axioms Proofs.LipschitzCertDemo.certSF10_41
#print axioms Proofs.LipschitzCertDemo.certSF10_85
#print axioms Proofs.LipschitzCertDemo.certSF30_34
#print axioms Proofs.LipschitzCertDemo.certSF30_93
#print axioms Proofs.LipschitzCertDemo.certTF10_43
#print axioms Proofs.LipschitzCertDemo.certTF10_99
#print axioms Proofs.LipschitzCertDemo.certTF30_25
#print axioms Proofs.LipschitzCertDemo.certTF30_71
#print axioms Proofs.LipschitzCertDemo.cappedFullCerts10_certified
#print axioms Proofs.LipschitzCertDemo.cappedFullCerts30_certified
#print axioms Proofs.LipschitzCertDemo.unconFullCerts10_certified
#print axioms Proofs.LipschitzCertDemo.unconFullCerts30_certified
#print axioms Proofs.LipschitzCertDemo.scorecardFull

-- Per-pair LipSDP on the FULL-INPUT nets (LipschitzCertScorecardSDPFull{,Uncon}.lean):
-- the tighter-constant pass at 784-dim input, both radii. Capped σ≤2:
-- 92→93/100 @ ε=0.1 — EQUAL to the L2-PGD attack bound, the cert ≤ TRUE ≤ PGD
-- sandwich CLOSED — and 72→91/100 @ ε=0.3 (PGD 92); unconstrained: 76→91 @0.1
-- (PGD 94), 2→77 @0.3 (PGD 86). PSD witnesses: exact rational LDLᵀ column
-- squares, one linarith goal per pair (the pooled files' recipe — MEASURED
-- faster than the entrywise lipsdp_slack_of_cert route at both widths; the
-- exact-LDL fractions hurt 512 separate norm_num goals far more than one
-- linarith call). Spot-check: one pair chain (slack + squared bound), a
-- reverse-order wrapper, first/middle/last per-image certs at both radii,
-- and the aggregates.
#print axioms Proofs.LipschitzCertDemo.hS01SF
#print axioms Proofs.LipschitzCertDemo.pairSqSF_0_1
#print axioms Proofs.LipschitzCertDemo.pairSqSF_1_0
#print axioms Proofs.LipschitzCertDemo.certifiedSSF10_0
#print axioms Proofs.LipschitzCertDemo.certifiedSSF30_41
#print axioms Proofs.LipschitzCertDemo.certifiedSSF10_86
#print axioms Proofs.LipschitzCertDemo.certifiedSTF10_31
#print axioms Proofs.LipschitzCertDemo.certifiedSTF10_81
#print axioms Proofs.LipschitzCertDemo.sdpCappedFullCerts10_certified
#print axioms Proofs.LipschitzCertDemo.sdpCappedFullCerts30_certified
#print axioms Proofs.LipschitzCertDemo.sdpUnconFullCerts10_certified
#print axioms Proofs.LipschitzCertDemo.sdpUnconFullCerts30_certified
#print axioms Proofs.LipschitzCertDemo.scorecard_sdp_full
#print axioms Proofs.LipschitzCertDemo.scorecard_sdp_full_uncon

-- IBP L∞ scorecard (IntervalBound.lean + LipschitzCertScorecardIBP{,Uncon}.lean):
-- the third certificate axis — exact interval bound propagation, pixel-L∞
-- ε ∈ {1,2,4,8}/255, same full-input nets. Capped σ≤2: 92/88/69/24 per 100
-- (PGD-L∞ 93/93/92/88; the L2 Lipschitz cert via ‖δ‖₂ ≤ √784·ε∞ manages only
-- 92/85/49/2 — the box beats the ball); uncon 87/42/2/0. Sign-split dense
-- boxes + endpoint-max ReLU, LINEAR in width; layer 1 = uniform box, reusing
-- the dotZ hpre facts + one absSumZ kernel fact per row (ListDot.lean).
-- Spot-check: the core soundness chain, the ℓ1 bridge, one absSumZ row fact +
-- wrapper, first/last per-image certs across the ε grid, and the aggregates.
-- (core, audited in AuditAxioms) Proofs.sum_getD_abs
-- (core, audited in AuditAxioms) Proofs.sum_getD_abs_div
-- (core, audited in AuditAxioms) Proofs.LipschitzCertDemo.denseLo_le
-- (core, audited in AuditAxioms) Proofs.LipschitzCertDemo.le_denseHi
-- (core, audited in AuditAxioms) Proofs.LipschitzCertDemo.relu_box
-- (core, audited in AuditAxioms) Proofs.LipschitzCertDemo.denseLo_uniform
-- (core, audited in AuditAxioms) Proofs.LipschitzCertDemo.denseLo2_eval
-- (core, audited in AuditAxioms) Proofs.LipschitzCertDemo.denseHi2_eval
-- (core, audited in AuditAxioms) Proofs.LipschitzCertDemo.ibp2_certified_at_eps
-- (azSF0 and the other raw absSumZ kernel facts are propext-ONLY — stricter
-- than the triple but they'd trip the exact-triple grep; audited transitively
-- via absrowSF below.)
#print axioms Proofs.LipschitzCertDemo.absrowSF
#print axioms Proofs.LipschitzCertDemo.absrowTF
#print axioms Proofs.LipschitzCertDemo.hbSFe1_0
#print axioms Proofs.LipschitzCertDemo.certIBPSFe1_0
#print axioms Proofs.LipschitzCertDemo.certIBPSFe8_0
#print axioms Proofs.LipschitzCertDemo.certIBPTFe1_0
#print axioms Proofs.LipschitzCertDemo.certIBPTFe2_91
#print axioms Proofs.LipschitzCertDemo.ibpCappedCertse1_certified
#print axioms Proofs.LipschitzCertDemo.ibpCappedCertse8_certified
#print axioms Proofs.LipschitzCertDemo.ibpUnconCertse1_certified
#print axioms Proofs.LipschitzCertDemo.scorecard_ibp
#print axioms Proofs.LipschitzCertDemo.scorecard_ibp_uncon
