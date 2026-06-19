# cifar8 optimizer ablation: {plain SGD, Nesterov momentum, AdamW} × {no-BN, BN} (2026-06-19)

Optimizer ablation on the deeper 8-conv CIFAR-10 CNN (`cifar8Verified` /
`cifar8BnVerified`; [16,16,32,32], 4 pools 32→2, → 128→64→64→10). All cells
share the proof-rendered forward + backward + parameter-gradient body — **only
the optimizer update op differs** (plain SGD `θ−lr·∇`, Nesterov momentum
`v←μv+∇; θ←θ−lr·(μv+∇)`, or AdamW). The Adam/momentum updates are
`ViTRender.emitAdamV` / `emitMomentum` swapped onto the same gradient; cotangent
divided by B (mean gradients), no label smoothing → a clean optimizer comparison.

## Setup

- Net: no-BN = 22 params / 52,858 floats; BN = 38 params (8× per-channel γ/β added)
- Schedule: 40 epochs each, batch=128, He init, random-hflip augment
- Plain SGD: flat lr (no schedule, no momentum) — the pre-existing baseline
- Momentum: μ=0.9 Nesterov, baseLR 0.02, 3-epoch warmup + cosine decay, no weight decay
- AdamW: lr 1e-3, β (.9,.999), wd 1e-4, 3-epoch warmup + cosine decay
- Backend: IREE + ROCm/HIP on RX 7900 XTX (gfx1100); runs share the GPU (pairwise concurrent)
- Renders: `tests/TestCifar8AdamTrain.lean` (adam + mom), `tests/RenderCifar8Sgd02.lean` (lr sweep)

## Results (test accuracy: final epoch / best)

| net | plain SGD (lr 0.1, flat) | Nesterov momentum (μ.9, lr.02) | AdamW (lr 1e-3) |
|---|---|---|---|
| no-BN | 66.72% / 68.36% | **76.99% / 77.20%** | 74.13% / 74.32% |
| BN    | 66.03% / 66.83% | **76.08% / 76.26%** | 73.52% / 73.74% |

Extra SGD point — no-BN flat lr=0.02: 65.00% / 65.71% (below lr=0.1).

Findings:
- **Nesterov momentum wins both rows** — ~+3 pts over AdamW, ~+10 pts over plain flat SGD.
  On a small vision net, well-tuned SGD+momentum edges out Adam's per-parameter adaptive
  scaling (a frequently-observed generalization gap).
- **The large plain-SGD↔Adam gap was the optimizer, not the lr.** Dropping SGD's lr to 0.02
  made it *worse* (65.0% < 66.7%), so the plain-SGD optimum was already ≥0.1; lr tuning alone
  never reaches Adam. Adding momentum is what closes (and reverses) the gap.
- **BN is ~neutral** here — a touch below no-BN under every optimizer — consistent with the
  earlier finding that BN's benefit for this net is init/lr-conditioning-sensitive.
- All seven runs converged cleanly (no NaN / loss spikes).

## Files

- `nobn_sgd.log` / `bn_sgd.log` — plain SGD, lr 0.1 (`cifar8{,-bn}-verified`)
- `nobn_sgd_lr02.log` — plain SGD, lr 0.02, no-BN (re-rendered train step; committed artifact restored after)
- `nobn_mom.log` / `bn_mom.log` — Nesterov momentum (`cifar8{,-bn}-verified-momentum`)
- `nobn_adam.log` / `bn_adam.log` — AdamW (`cifar8{,-bn}-verified-adam`)
