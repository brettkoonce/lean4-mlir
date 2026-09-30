# 2026-09-30 — PGD demos with a sound certified radius (rubric_review WP3)

The gate runs for WP3 of `planning/rubric_review.md`. Before this change, the `mnist-*-pgd`
demos' "certified-robust acc" used a power-iteration `L`, which is a lower bound on each ‖Wᵢ‖₂. So
the printed radius `m/(√2·L)` was not an instance of `lipschitz_margin_certified_radius`. Each
layer's `L` is now the Schatten-8 upper bound that `denseE_lipschitzL2_gram2` proves, computed on
the host with a relative rounding slack of 1e-6. The conv factors are tap-sums of per-tap
Schatten-8 bounds. The power-iteration value is still printed, as the estimate.

One 4060 Ti (`CUDA_VISIBLE_DEVICES=0`), default configs except the CNN (`CNN_PGD_EPOCHS=1`, a
smoke):

| run | L (bound) | product of estimates | cert L2 ε=0.5 / 1.0 / 1.5 | L2 PGD ε=0.5 / 1.0 / 1.5 |
|---|---|---|---|---|
| `mnist-linear-pgd` (`linear.log`) | 5.744 | 5.293 | 48.10 / 5.17 / 0.17 % | 79.38 / 52.27 / 20.14 % |
| `mnist-mlp-pgd` (`mlp.log`) | 56.82 | 34.06 | 0 / 0 / 0 % | 87.67 / 52.24 / 15.33 % |
| `mnist-cnn-pgd`, 1 epoch (`cnn_1ep.log`) | 1383 | 443.8 | 0 / 0 / 0 % | 86.86 / 75.83 / 59.76 % |

In every row the certified accuracy is at or below L2 PGD at the same ε, as the sandwich requires.
Two caveats remain:
- The MLP and CNN certificates were vacuous at these radii before the change as well.
- The certificate covers the real-arithmetic net at the margin the float forward printed. The f32
  rounding of the logits is not budgeted (see `LeanMlir/Verified/Attack.lean`'s module doc).

`conv1→32` has a single input channel, so each tap matrix has rank 1 and its Schatten-8 bound
equals σ₁: bound and estimate agree there.
