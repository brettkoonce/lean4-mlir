# MNIST DDPM, scored under ancestral sampling

The demo's default sampler moved from deterministic DDIM (η = 0) to ancestral sampling (η = 1),
the best row of the book's η sweep at every budget. These are the two 50-epoch checkpoints of
`runs/2026-08-28-mnist-ddpm-verified-score/` scored under it, same scorer, same 1024 samples,
same 50 steps. Both checkpoints are unchanged since 2026-08-28.

    lake exe mnist-ddpm-score 1024 50          # centred arm, η = 1 (now the default)
    lake exe mnist-ddpm-score 1024 50 raw      # uncentred arm
    python3 scripts/demos/mnist_ddpm_score.py

| arm | sampler | coverage | confidence | energy | × floor |
|---|---|---|---|---|---|
| 50 epochs, centred | ancestral, η = 1 | 10/10 | 92.98% | 0.00669 | 4× |
| 50 epochs, uncentred | ancestral, η = 1 | 10/10 | 93.60% | 0.00959 | 6× |
| 50 epochs, centred | deterministic, η = 0 | 10/10 | 90.94% | 0.02365 | 15× |
| 50 epochs, uncentred | deterministic, η = 0 | 9/10 | 92.82% | 0.05255 | 33× |

The η = 0 rows were re-measured with the refactored sampler (`Ddpm.ddimCoefs`) and match the
2026-08-28 numbers to every printed digit; the coefficients are bitwise identical to the old inline
forms over the whole 50-step schedule at η ∈ {0, 0.25, 0.5, 1}. The η = 1 centred row equals the
book's 0.0067 at NFE 50.

`generate_*.log` is the Lean driver, `score_*.log` the statistics.
