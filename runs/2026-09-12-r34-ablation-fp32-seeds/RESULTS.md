# ResNet-34 / Imagenette recipe ablation — fp32, mean ± 95% CI over seeds

Directory `runs/2026-09-12-r34-ablation-fp32-seeds`. Interval = t_(0.975, n−1)·sd/√n. Δ intervals: paired by seed (shared init + data order) and unpaired (Welch).

| arm | n | seeds | top-1 mean | sd | ±95% | Δ vs full | ±95% paired | ±95% unpaired |
|---|---|---|---|---|---|---|---|---|
| full | 5 | 1,2,3,4,5 | 89.99 | 0.26 | ±0.32 | +0.00 | ±0.00 | ±0.00 |
| nowd | 5 | 1,2,3,4,5 | 89.98 | 0.35 | ±0.44 | -0.02 | ±0.61 | ±0.46 |
| nowarm | 5 | 1,2,3,4,5 | 89.44 | 0.14 | ±0.17 | -0.55 | ±0.46 | ±0.32 |
| nols | 5 | 1,2,3,4,5 | 89.30 | 0.23 | ±0.29 | -0.69 | ±0.42 | ±0.36 |
| noadam | 5 | 1,2,3,4,5 | 87.72 | 0.27 | ±0.34 | -2.27 | ±0.59 | ±0.39 |
| nocos | 5 | 1,2,3,4,5 | 84.99 | 1.93 | ±2.40 | -5.00 | ±2.60 | ±2.42 |
| noaug | 5 | 1,2,3,4,5 | 82.66 | 0.32 | ±0.39 | -7.33 | ±0.49 | ±0.42 |
| bare | 5 | 1,2,3,4,5 | 83.20 | 1.27 | ±1.57 | -6.79 | ±1.86 | ±1.61 |
| sgd10 | 5 | 1,2,3,4,5 | 88.13 | 0.29 | ±0.36 | -1.86 | ±0.57 | ±0.40 |

## pgfplots (Δ vs full, paired 95% half-width as the x error bar)

```
\addplot[only marks, mark=*, error bars/.cd, x dir=both, x explicit] coordinates {
  (0.00,full) +- (0.00,0) (-0.02,nowd) +- (0.61,0) (-0.55,nowarm) +- (0.46,0) (-0.69,nols) +- (0.42,0) (-2.27,noadam) +- (0.59,0) (-5.00,nocos) +- (2.60,0) (-7.33,noaug) +- (0.49,0) (-6.79,bare) +- (1.86,0) (-1.86,sgd10) +- (0.57,0)
};
```
