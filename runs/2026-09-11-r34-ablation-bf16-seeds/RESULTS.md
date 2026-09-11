# ResNet-34 / Imagenette recipe ablation — bf16, mean ± 95% CI over seeds

Directory `runs/2026-09-11-r34-ablation-bf16-seeds`. Interval = t_(0.975, n−1)·sd/√n. Δ intervals: paired by seed (shared init + data order) and unpaired (Welch).

| arm | n | seeds | top-1 mean | sd | ±95% | Δ vs full | ±95% paired | ±95% unpaired |
|---|---|---|---|---|---|---|---|---|
| full | 5 | 1,2,3,4,5 | 89.98 | 0.12 | ±0.15 | +0.00 | ±0.00 | ±0.00 |
| nowd | 5 | 1,2,3,4,5 | 90.12 | 0.45 | ±0.56 | +0.14 | ±0.45 | ±0.53 |
| nowarm | 5 | 1,2,3,4,5 | 89.17 | 0.45 | ±0.56 | -0.82 | ±0.62 | ±0.54 |
| nols | 5 | 1,2,3,4,5 | 89.48 | 0.45 | ±0.55 | -0.50 | ±0.62 | ±0.53 |
| noadam | 5 | 1,2,3,4,5 | 87.77 | 0.28 | ±0.35 | -2.21 | ±0.33 | ±0.35 |
| nocos | 5 | 1,2,3,4,5 | 85.20 | 2.18 | ±2.71 | -4.78 | ±2.74 | ±2.71 |
| noaug | 5 | 1,2,3,4,5 | 82.87 | 0.33 | ±0.42 | -7.11 | ±0.38 | ±0.41 |
| bare | 5 | 1,2,3,4,5 | 83.85 | 0.33 | ±0.41 | -6.14 | ±0.36 | ±0.40 |

## pgfplots (Δ vs full, paired 95% half-width as the x error bar)

```
\addplot[only marks, mark=*, error bars/.cd, x dir=both, x explicit] coordinates {
  (0.00,full) +- (0.00,0) (0.14,nowd) +- (0.45,0) (-0.82,nowarm) +- (0.62,0) (-0.50,nols) +- (0.62,0) (-2.21,noadam) +- (0.33,0) (-4.78,nocos) +- (2.74,0) (-7.11,noaug) +- (0.38,0) (-6.14,bare) +- (0.36,0)
};
```
