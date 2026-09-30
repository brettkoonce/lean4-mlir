# Certificate scorecards, re-measured (2026-09-30)

The certificate-tier Lean files state only the per-image certificates they emit (e.g.
`scorecard_sdp : sdpCappedCerts.length = 8 ∧ …`). The dataset-level counts — how many of the first
100 MNIST test images each method certifies — are measurements the generators print, and until
2026-09-30 they lived in the generated files' comments. The comment-numbers rule
(`planning/rubric_review.md`, decision 6) moved them here.

Each generator was re-run from the tree at `bee5ab0d` plus the staged WP6 comment edits
(`.head`), with MNIST in `data/`. Every emitted Lean file came back byte-identical to the
committed one, so these counts are the ones the committed certificates were cut from. Logs:
one `<generator>.log` per script, with `/usr/bin/time -v` at the end.

All counts are over the first 100 MNIST test images. PGD is the attack's empirical robust count,
an upper bracket on what any certificate can reach.

## Pooled 49-dim MLP, L2, ε = 1/10

`lipschitz_cert_scorecard.py` (global √2·∏‖Wᵢ‖ criterion) and `lipschitz_cert_pair_sdp.py`
(per-pair LipSDP constants).

| net | quantized test acc | global certificate | LipSDP | PGD |
|---|---|---|---|---|
| capped (`mlpS`, /256) | 0.8703 | 34/100 | 69/100 | 72/100 |
| unconstrained (`mlpT`, /128) | 0.8983 | 1/100 | 63/100 | 69/100 |

## Full 784-dim MLP, L2

`lipschitz_cert_scorecard_full.py` and `lipschitz_cert_pair_sdp_full.py`.

| net | quantized test acc | L | ε | global certificate | LipSDP | PGD |
|---|---|---|---|---|---|---|
| capped (`SF`, σ ≤ 2) | 0.9238 | 4.953 | 0.1 | 92/100 | 93/100 | 93/100 |
| | | | 0.3 | 72/100 | 91/100 | 92/100 |
| unconstrained (`TF`) | 0.9508 | 29.847 | 0.1 | 76/100 | 91/100 | 94/100 |
| | | | 0.3 | 2/100 | 77/100 | 86/100 |

## Full 784-dim MLP, pixel L∞ at ε = 1, 2, 4, 8 /255

`lipschitz_cert_scorecard_ibp.py` (interval bound propagation) and `crown_ibp_scorecard.py`
(CROWN-IBP). "L2-implied" is what the L2 certificate gives at the equivalent radius.

| net | IBP | CROWN-IBP | L2-implied | PGD-L∞ |
|---|---|---|---|---|
| capped (`SF`) | 92 / 88 / 69 / 24 | 93 / 93 / 92 / 81 | 92 / 85 / 49 / 2 | 93 / 93 / 92 / 88 |
| unconstrained (`TF`) | 87 / 42 / 2 / 0 | 94 / 92 / 76 / 15 | 71 / 14 / 0 / 0 | 95 / 92 / 85 / 36 |

## Not re-run

- `ibp_conv_scorecard.py` (the conv-net IBP tier, `IbpConvScorecard/*`) crashes at
  `OUT.with_name(...)` and its paths predate the certificate-directory layout; its files were
  hand-matched in WP6c. Its last recorded counts (79 / 73 / 47 / 13 at ε = 1, 2, 4, 8 /255) are in
  `historical/comment_measurements.md`.
- The randomized-smoothing scorecards keep their inputs in `runs/2026-07-12-smooth-scorecard/`;
  `smooth_scorecard_gen.py --check` and `smooth_dec_scorecard_gen.py --check` pass.
