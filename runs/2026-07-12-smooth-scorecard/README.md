# Smoothing CP scorecard runs (2026-07-12)

Fixed-protocol randomized-smoothing certification runs — the data behind the
generated `LeanMlir/Proofs/SmoothingCPScorecard.lean` corpus.

- **Protocol**: first-100 test images (`SMOOTH_STRIDE=1`, `SMOOTH_MAXCERT=100`),
  σ = 0.5 only (`SMOOTH_SIGMA_MILLI=500`), n = 10112 estimation samples
  (default `SMOOTH_N=10000` rounded up to 79×128), n₀ = 128, α = 0.001.
- **Launcher**: `run_smooth_scorecard.sh` (2× gfx1100; GPU0 cifar,
  GPU1 cnn→mlp; ~25 min wall).
- **Exes**: `mnist-mlp-smooth`, `mnist-cnn-smooth`, `cifar-smooth`
  (`VerifiedNet.smoothCertify`, proof-rendered fwd on IREE/rocm).
- **CSV columns**: `sigma,img_idx,label,pred,abstain,radius,count,n` — the
  `count,n` pair is what the Lean kernel tail checks consume.

| net | clean acc (noise-trained) | certified (first-100) | of which correct | ACR |
|---|---|---|---|---|
| MNIST-MLP | 97.35% | 99/100 | 99 | 1.154 |
| MNIST-CNN | 98.01% | 100/100 | 100 | 1.320 |
| CIFAR-CNN | 54.54% | 80/100 | 60 | 0.390 |

Downstream: `scripts/smooth_scorecard_gen.py` reads
`runs/smooth_<slug>_scorecard.csv` (same files as here), binary-searches the
largest 4-decimal `q₀ = a/10000` with `binomTail n k q₀ ≤ α` (exact integer
arithmetic), and emits one `binomTail_le_of_kernel_check` + `decide +kernel`
lemma per certified image.
