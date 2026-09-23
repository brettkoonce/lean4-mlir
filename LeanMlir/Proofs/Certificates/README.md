# Certificates/ — machine-emitted scorecards over hand-written engines

The scorecard, instance and witness files in this directory are **generated**
by a script in `scripts/` (`lipschitz_cert_*.py`, `crown_ibp_*.py`,
`ibp_conv_scorecard.py`, `smooth_*scorecard*_gen.py`,
`smoothing_net_witness_gen.py`) and then checked by Lean like any other
proof: same kernel, zero `sorry`s, three-axiom audit. Ten files are
hand-written — the engines the generated files instantiate and the trained
weight instance: `LipschitzCert.lean`, `DenseEuclid.lean` (the dense / ReLU layers, their L2
bounds and `CertifiedAt`), `LipschitzCertInstance.lean`, `LipschitzCertPairSDP.lean`,
`GaussianQuantile.lean` (Φ, Φ⁻¹, full support), `SmoothingCP.lean`, `SmoothingGaussian.lean`,
`SmoothingMC.lean`, `SmoothingNetSemantics.lean` and `SmoothingPhiBounds.lean`.

That provenance explains their shape: thousands of short, structurally
identical theorem statements (one per network / image / radius row of a
scorecard), high body duplication, thin docstrings. Judged as *prose* they
look nothing like the hand-written library in the sibling directories —
and that's expected; they are certified *data*, not exposition. If a
code-quality metric flags this directory, the honest answer is "yes, it's
an emitted payload, and here is the emitter."

To regenerate a scorecard, run its generator script from the repo root and
rebuild `lake build CertsHeavy` (the heavy scorecards) or `Certs` (the
rest). Don't hand-edit a generated file — edits will be clobbered by the next
generator run.

## The families, and where each piece lives

Every family has the same shape: the data is ℚ/ℤ/ℕ, the ℝ object is *defined* as its cast, one
computable check runs in the kernel, and one soundness lemma turns the check into an ℝ theorem.
The engines — the ℝ theorems and checkers — are hand-written; the data files are generated. Some
engines live in `Foundation/` (`IntervalBound`, `IntervalBoundConv`, `IntervalBoundConvQ`,
`CrownBound`, `GramQ`, `ListDot`).

| family | ℝ theorem (engine) | check → ℝ bridge | data (generated) | generator (`scripts/`) |
|---|---|---|---|---|
| L2, pooled 49-d | `lipschitz_margin_certified_radius`, `certified_at_eps` (`DenseEuclid`) | per-entry `simp; norm_num`; Gram via `gram_eq_of_check` | `LipschitzCertInstance` (data half, hand-merged), `LipschitzCertScorecard` | `lipschitz_cert_{rationalize,power_iter,witness_s8}.py` (snippets to merge), `lipschitz_cert_scorecard.py` |
| L2, full 784-d | same | `ListDot.dotZ` | `LipschitzCertScorecardFull{,Nets,ImgsA,ImgsB}` | `lipschitz_cert_scorecard_full.py` |
| LipSDP | `pair_sq_bound`, `certified_at_eps_pair` | `linarith` over LDLᵀ column squares | `LipschitzCertScorecardSDP{,Uncon}`; `…SDPFull{,Uncon}` (built by no lib — OOM) | `lipschitz_cert_pair_sdp{,_full}.py` |
| float tier | `FloatBridge` | — | `LipschitzCertFloat` | `lipschitz_cert_float.py` |
| IBP L∞, dense | `ibp2_certified_at_eps` | `dotZ` / `absSumZ` | `LipschitzCertScorecardIBP{Data,,Uncon}` | `lipschitz_cert_scorecard_ibp.py` |
| CROWN L∞ | `crown2_certified_at_eps` | `combZ` + per-entry `simp` | `LipschitzCertScorecardCrown{,Uncon}` | `crown_ibp_scorecard.py` |
| IBP L∞, conv | `ibp3_certified_of_boxSound` | ℚ checker `convNetCheckQ_sound` | `IbpConvScorecard{Net,ImgsA–D,}` | `ibp_conv_scorecard.py` |
| smoothing, CP | `smoothing_cp_certified_solved` | ℕ binomial tail, `binomTail_le_of_kernel_check` | `SmoothingCPScorecard` | `smooth_scorecard_gen.py` |
| smoothing, decimal radii | `smooth_radius_dec` | ℚ Φ scan | `SmoothingDecChunk1–6`, `SmoothingDecScorecard` | `smooth_dec_scorecard_gen.py` |
| smoothing ↔ net | `smoothing_cp_certified_mlpT` | per-witness `simp` | `SmoothingNetWitness` | `smoothing_net_witness_gen.py` |

⚠ The Lipschitz / SDP / IBP / CROWN scorecards end in `CertifiedAt*` theorems about the net. The
smoothing CP and decimal scorecards end in their side conditions (`binomTail … ≤ α`,
`m/2000 ≤ σΦ⁻¹(q₀)`); only the pooled `mlpT` composes them with a net theorem.

**Names.** `LipschitzCertDemo` (namespace) and the `LipschitzCert*` file prefix also cover the L∞
IBP / CROWN scorecards. Capped vs unconstrained is `S`/`T` in net names (`mlpS`/`mlpT`), `C`/`U`
in pooled theorems, `SC`/`SU` in SDP, and nothing / `Uncon` in file names. `F` = full 784-d input,
`e8` = ε = 8/255, `ImgsA–D` = image chunks split for memory.
