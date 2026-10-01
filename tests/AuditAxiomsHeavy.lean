import LeanMlir.Proofs.Certificates.LipschitzCert.ScorecardFull
-- (LipschitzCertScorecardSDPFull{,Uncon} imports DISABLED with their lib membership — the
-- linarith PSD witnesses OOM the free-tier runners. Their prints below are commented out
-- with them; re-enable both together. Until then `scripts/certs/check_sdpfull.sh` builds
-- both and audits every theorem in them, locally.)
import LeanMlir.Proofs.Certificates.LipschitzCert.ScorecardIBP
import LeanMlir.Proofs.Certificates.LipschitzCert.ScorecardIBPUncon
import LeanMlir.Proofs.Certificates.LipschitzCert.ScorecardCrown
import LeanMlir.Proofs.Certificates.LipschitzCert.ScorecardCrownUncon
import LeanMlir.Proofs.Certificates.IbpConvScorecard.Basic

/-! # Axiom audit — the HEAVY generated certificate corpus (`CertsHeavy`)

The full-input (784-dim) scorecard instances: L2 Tsuzuku, dense IBP L∞, CROWN and the conv
IBP net (the per-pair LipSDP files are in no lib; their lines below are commented out), generated
weight/image data and per-image theorems. Split out
of `tests/AuditAxioms.lean` together with the `Certs`→`CertsHeavy` lakefile
split: the long-running data-heavy corpus gets its own workflow
(.github/workflows/certs-heavy.yml) so it cannot take certs.yml/blueprint.yml
down. Same gate: every line below must close under exactly
`[propext, Classical.choice, Quot.sound]`. The hand-written engine cores
(ListDot.lean, IntervalBound.lean) stay in `Certs` and are audited by the MAIN
audit. -/

-- FULL-INPUT scorecard (LipschitzCert/ScorecardFull*.lean): the pooled 49-dim reduction
-- lifted to the genuine 784-dim input (exact k/255 pixels), per-image certificates at
-- pixel-L2 ε = 1/10 AND 3/10 on two 784→16→10 nets, capped σ≤2 and unconstrained (the
-- measured counts are in the generated header). Engine: ListDot.lean
-- — every 784-term dot is one kernel `dotZ` evaluation (`decide +kernel`, GMP,
-- propext-only; NOT native_decide) transported to the `Fin 784` sums by the once-proved
-- `sum_getD_div` bridge; the pooled recipe's simp sum walk is quadratic in input dim and
-- priced out at 784.
-- Spot-check: the bridge core, both nets' Schatten-8 chains, a Gram entry +
-- wrapper per net, first/middle/last per-image certs at both radii, and the
-- mechanized aggregates. (The raw `gz*` kernel dot facts are propext-ONLY —
-- stricter than the triple — so they'd trip the exact-triple CI grep; they're
-- covered transitively by every entry lemma, e.g. gSF_0_15 below.)
-- (core, audited in AuditAxioms) Proofs.dotZ_comm
-- (core, audited in AuditAxioms) Proofs.sum_getD_mul
-- (core, audited in AuditAxioms) Proofs.sum_getD_div
#print axioms Proofs.Robustness.gSF_0_15
#print axioms Proofs.Robustness.G1SF_eq
#print axioms Proofs.Robustness.H1SF_eq
#print axioms Proofs.Robustness.W1SF_lip
#print axioms Proofs.Robustness.W2SF_lip
#print axioms Proofs.Robustness.mlpSF_lip
#print axioms Proofs.Robustness.G1TF_eq
#print axioms Proofs.Robustness.W1TF_lip
#print axioms Proofs.Robustness.mlpTF_lip
#print axioms Proofs.Robustness.hpreSF0_eval
#print axioms Proofs.Robustness.marginSF0
#print axioms Proofs.Robustness.certSF10_0
#print axioms Proofs.Robustness.certSF10_4
#print axioms Proofs.Robustness.certSF10_7
#print axioms Proofs.Robustness.certSF30_0
#print axioms Proofs.Robustness.certSF30_4
#print axioms Proofs.Robustness.certSF30_7
#print axioms Proofs.Robustness.certTF10_0
#print axioms Proofs.Robustness.certTF10_5
#print axioms Proofs.Robustness.certTF10_9
#print axioms Proofs.Robustness.certTF30_25
#print axioms Proofs.Robustness.certTF30_71
#print axioms Proofs.Robustness.cappedFullCerts10_certified
#print axioms Proofs.Robustness.cappedFullCerts30_certified
#print axioms Proofs.Robustness.unconFullCerts10_certified
#print axioms Proofs.Robustness.unconFullCerts30_certified
#print axioms Proofs.Robustness.scorecardFull

-- Per-pair LipSDP on the FULL-INPUT nets (LipschitzCertScorecardSDPFull{,Uncon}.lean):
-- the tighter-constant pass at 784-dim input, both radii (measured counts against the PGD
-- bracket: the generated header). PSD witnesses: exact rational LDLᵀ column squares, one linarith goal per pair
-- (the pooled files' recipe — MEASURED faster than an entrywise norm_num check at both
-- widths; the exact-LDL fractions hurt 512 separate norm_num goals far more than one
-- linarith call). Those counts are exact-rational MEASUREMENTS; the first 8 certifying
-- images per radius carry the `CertifiedAt` theorems, and the `scorecard_sdp_full*`
-- aggregates state only those.
-- Spot-check: one pair chain (slack + squared bound), a reverse-order wrapper,
-- first/middle/last per-image certs at both radii, and the aggregates.
-- #print axioms Proofs.Robustness.hS01SF  -- CI-disabled with the SDP lib membership
-- #print axioms Proofs.Robustness.pairSqSF_0_1  -- CI-disabled with the SDP lib membership
-- #print axioms Proofs.Robustness.pairSqSF_1_0  -- CI-disabled with the SDP lib membership
-- #print axioms Proofs.Robustness.certifiedSSF10_0  -- CI-disabled with the SDP lib membership
-- #print axioms Proofs.Robustness.certifiedSSF30_3  -- CI-disabled with the SDP lib membership
-- #print axioms Proofs.Robustness.certifiedSSF10_7  -- CI-disabled with the SDP lib membership
-- #print axioms Proofs.Robustness.certifiedSTF10_3  -- CI-disabled with the SDP lib membership
-- #print axioms Proofs.Robustness.certifiedSTF30_7  -- CI-disabled with the SDP lib membership
-- #print axioms Proofs.Robustness.sdpCappedFullCerts10_certified  -- CI-disabled with the SDP lib membership
-- #print axioms Proofs.Robustness.sdpCappedFullCerts30_certified  -- CI-disabled with the SDP lib membership
-- #print axioms Proofs.Robustness.sdpUnconFullCerts10_certified  -- CI-disabled with the SDP lib membership
-- #print axioms Proofs.Robustness.sdpUnconFullCerts30_certified  -- CI-disabled with the SDP lib membership
-- #print axioms Proofs.Robustness.scorecard_sdp_full  -- CI-disabled with the SDP lib membership
-- #print axioms Proofs.Robustness.scorecard_sdp_full_uncon  -- CI-disabled with the SDP lib membership

-- IBP L∞ scorecard (IntervalBound.lean + LipschitzCertScorecardIBP{,Uncon}.lean):
-- the third certificate axis — exact interval bound propagation, pixel-L∞
-- ε ∈ {1,2,4,8}/255, same full-input nets (measured counts against the PGD-L∞ bracket and
-- the L2 cert via ‖δ‖₂ ≤ √784·ε∞: the generated header). Sign-split dense
-- boxes + endpoint-max ReLU, LINEAR in width; layer 1 = uniform box, reusing
-- the dotZ hpre facts + one absSumZ kernel fact per row (ListDot.lean).
-- Spot-check: the core soundness chain, the ℓ1 bridge, one absSumZ row fact +
-- wrapper, first/last per-image certs across the ε grid, and the aggregates.
-- (core, audited in AuditAxioms) Proofs.sum_getD_abs
-- (core, audited in AuditAxioms) Proofs.sum_getD_abs_div
-- (core, audited in AuditAxioms) Proofs.Robustness.denseLo_le
-- (core, audited in AuditAxioms) Proofs.Robustness.le_denseHi
-- (core, audited in AuditAxioms) Proofs.Robustness.relu_box
-- (core, audited in AuditAxioms) Proofs.Robustness.denseLo_uniform
-- (core, audited in AuditAxioms) Proofs.Robustness.denseLo2_eval
-- (core, audited in AuditAxioms) Proofs.Robustness.denseHi2_eval
-- (core, audited in AuditAxioms) Proofs.Robustness.ibp2_certified_at_eps
-- (azSF0 and the other raw absSumZ kernel facts are propext-ONLY — stricter
-- than the triple but they'd trip the exact-triple grep; audited transitively
-- via absrowSF below.)
#print axioms Proofs.Robustness.absrowSF
#print axioms Proofs.Robustness.absrowTF
#print axioms Proofs.Robustness.hbSFe1_0
#print axioms Proofs.Robustness.certIBPSFe1_0
#print axioms Proofs.Robustness.certIBPSFe8_0
#print axioms Proofs.Robustness.certIBPTFe1_0
#print axioms Proofs.Robustness.certIBPTFe2_4
#print axioms Proofs.Robustness.ibpCappedCertse1_certified
#print axioms Proofs.Robustness.ibpCappedCertse8_certified
#print axioms Proofs.Robustness.ibpUnconCertse1_certified

-- CROWN, the SAME nets/subset/ε grid as the IBP tier above — a new COLUMN in
-- that table (measured counts: the generated header). Each emitted image is proved at the
-- LARGEST radius it is emitted at and carried down the grid by
-- `CertifiedAtLinf.mono`, so the spot-checks below cover both a directly-proved
-- certificate (e8/e4) and a mono-derived one (e1). The `nrm*` raw
-- `absSumZ (combZ …)` kernel facts are propext-ONLY (stricter than the triple,
-- but they would trip the exact-triple grep); they are audited transitively via
-- the `hl1*` wrappers and the certificates below.
#print axioms Proofs.Robustness.hWSF
#print axioms Proofs.Robustness.hWTF
#print axioms Proofs.Robustness.hl1SFe8_0_0
#print axioms Proofs.Robustness.hrelSFe8_0
#print axioms Proofs.Robustness.hcertSFe8_0
#print axioms Proofs.Robustness.certCRSFe8_0
#print axioms Proofs.Robustness.certCRSFe1_0
#print axioms Proofs.Robustness.certCRSFe8_7
#print axioms Proofs.Robustness.certCRTFe8_3
#print axioms Proofs.Robustness.certCRTFe4_0
#print axioms Proofs.Robustness.crownCappedCertse1_certified
#print axioms Proofs.Robustness.crownCappedCertse8_certified
#print axioms Proofs.Robustness.crownUnconCertse1_certified
#print axioms Proofs.Robustness.scorecard_crown
#print axioms Proofs.Robustness.scorecard_crown_uncon
#print axioms Proofs.Robustness.scorecard_ibp
#print axioms Proofs.Robustness.scorecard_ibp_uncon

-- CONVOLUTIONAL IBP instance (Certificates/IbpConvScorecard/Basic.lean, engine
-- Proofs.Certificates.IntervalBoundConv.Basic): the first certificate in the repo covering a
-- convolution, a max-pool, and more than two layers — `conv2d(1→4, 3×3 SAME) → reluT
-- → maxPool2 → denseT(64→10)` at trained k/256 weights, on 8×8 4×4-pooled MNIST,
-- pixel-L∞ ε ∈ {1,2,4,8}/255 (the proved images are the aggregates' witness lists; the
-- measured count comes from `scripts/certs/ibp_conv_scorecard.py`). Per image the
-- box is checked ONCE, at the largest certifying radius, by a `decide +kernel` of the
-- exact-ℚ checker (Proofs.Certificates.IntervalBoundConv.Q); the smaller radii are
-- `CertifiedAtLinf3.mono` corollaries. Spot-check: the net's box-soundness chain, the
-- checker's soundness and its instance, and the four aggregates.
#print axioms Proofs.IBP.ConvNet.net_boxSound
#print axioms Proofs.IBP.convNetCheckQ_sound
#print axioms Proofs.IBP.ConvNet.net_certified_of_check
#print axioms Proofs.IBP.ConvNet.certsE1_certified
#print axioms Proofs.IBP.ConvNet.certsE2_certified
#print axioms Proofs.IBP.ConvNet.certsE4_certified
#print axioms Proofs.IBP.ConvNet.certsE8_certified
#print axioms Proofs.IBP.ConvNet.scorecard_ibp_conv
