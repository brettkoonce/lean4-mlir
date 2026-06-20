# cifar8-wide (d1=512, MNIST-style head) optimizer ablation (2026-06-20)

The wide-head peer of `runs/ablation_cifar8/` — the **bridge** net of Chapter 5: the same
8-conv backbone, but the dense head swapped to MNIST's 2×512 (`128→512→512→10`, d1=512)
instead of the narrow `128→64→64→10`. Same controlled pipeline as the narrow ablation
(per-epoch shuffle + random hflip + cosine-warmup), only the optimizer (and BN) varies.
Net: 373,626 floats (vs 52,858 narrow); ~89% of params in the head.

## Results — d1=512, modern pipeline (final % / best %)

| net | SGD (lr 0.1) | Nesterov momentum (μ.9, lr.02) | AdamW (lr 1e-3) |
|---|---|---|---|
| no-BN | 72.39 / 72.42 | 76.70 / 76.85 | 73.57 / 73.70 |
| BN    | 74.23 / 74.28 | **77.08 / 77.08** | 73.35 / 73.42 |

Findings:
- **Momentum wins** (+3–4 over SGD); best on the board = BN+momentum 77.1%.
- **Head width barely matters**: 7.1× the params of the narrow net (25× the head), accuracy
  within a point (best 77.1 wide vs 77.2 narrow), and only ~1.36× wall-clock per epoch
  (≈14.3 vs ≈10.5 s, solo gfx1100) — the conv backbone dominates FLOPs, the head is cheap
  matmul. The depth, not the head, is the lever. This is the Ch5 "bridge" point.
- **BN helps most under plain SGD** (≈+1.8) and its margin shrinks to ~nil under momentum/Adam.

## Cosmetic loss=NaN

Some runs print `loss=NaN` in late epochs (e.g. no-BN momentum, ep40). This is a **display
artifact only**: the in-graph CE loss uses `onehot·log(softmax)` without a log-sum-exp trick,
so the wider net's larger logits let an off-class softmax underflow to 0 → `0·log(0)=NaN` in
the loss scalar. The cotangent (`softmax−onehot`, no log) and eval (argmax, no exp) are
unaffected — the eval accuracies above are healthy. Training is not corrupted.

## Files

- `nobn.log` — `cifar8w-ablation` (no-BN), SGD → momentum → AdamW in sequence
- `bn.log`   — `cifar8w-bn-ablation` (per-channel BN), same three optimizers

Render: `tests/TestCifar8WideTrain.lean` → `verified_mlir/cifar8w*{train_step,fwd}.mlir`.
Specs: `cifar8w{,Bn}Verified` in `LeanMlir/VerifiedNets.lean`.
