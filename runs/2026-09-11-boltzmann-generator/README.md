# 2026-09-11 Boltzmann generator — planning/boltzmann_generator_demo.md, phases 0–5

The 2-D diffusion exe (`demos/MainDiffusion2d.lean`, back out of archive/) with the
`muller_brown` target and the `flow` / `ot` / `reflow` arms. XLA backend, one GPU,
`LEAN_MLIR_MEM_FRACTION=0.1`, 18,178 params, batch 256, Adam 1e-3, 20,000 steps every arm.
Data: `python3 preprocess_boltzmann.py` (8 Langevin chains × 1e5 steps at kT = 20, burn 1e4,
thin 20, shuffled → 36,000 points; an independent 4,096-point draw from the quadrature density;
the 360×360 grid; one chain from B at kT = 12 and at 8). Scorer: `scripts/boltzmann_metrics.py`.

| arm | train | sample NFE 50 | notes |
|---|---|---|---|
| flow (independent coupling) | 29.7 s | ~1 s (+4 forwards/step for `logp`) | `flow_train.*`, `flow_nfe_sweep.*` |
| flow-ot (minibatch OT)      | 341 s  | | Hungarian at B = 256 is ~16 ms/step, 10× the gradient step |
| flow-reflow                 | 30.6 s | | 36,864 pairs drawn from the flow arm's ODE in 0.5 s |
| DDPM (cosine schedule)      | 34.0 s | | `ddpm_train.*`, `ddpm_nfe_sweep.*` |

## Table 1 — kT = 20, n = 2048 (`table1.txt`, `table1.json`)

```
  source                            p_A / p_B / p_C          <U>    dF_AB   energy (x floor)     KL(m||e) / KL(e||m)
  quadrature (exact)                0.806 / 0.129 / 0.065  -113.6    -36.6      1.0x (0.0015)
  Langevin, 8 chains x 1e5 steps    0.844 / 0.101 / 0.054  -112.6    -42.4      4.1x (0.0060)
  flow, Euler, NFE 50               0.834 / 0.103 / 0.063  -112.4    -41.9      3.2x (0.0046)    0.038 /  0.048
    ^ reweighted by p_20/p_theta    0.805 / 0.129 / 0.066  -112.8    -36.7                       ESS 1864
  flow, Heun, NFE 50 (25 steps)     0.834 / 0.109 / 0.056  -111.2    -40.6      2.4x (0.0035)
  flow, Euler, NFE 10               0.840 / 0.086 / 0.073  -117.7    -45.5      8.6x (0.0125)    0.152
  flow, Euler, NFE 2                0.891 / 0.006 / 0.103   -97.4    -98.9     90.8x (0.1320)    2.346
  DDPM, DDIM, NFE 50   [1 in wall]  0.895 / 0.059 / 0.046  -114.7    -54.4     18.0x (0.0261)
  DDPM, DDIM, NFE 10   [10 in wall] 0.841 / 0.098 / 0.061   -98.0    -43.0     13.5x (0.0196)
  DDPM, DDIM, NFE 2    [236]        0.742 / 0.144 / 0.115    83.3    -32.8    268.0x (0.3896)
  DDPM, Euler ODE, NFE 50 [61]      0.873 / 0.090 / 0.037   -50.4    -45.4     51.7x (0.0752)
  DDPM, Heun, NFE 50   [74]         0.865 / 0.101 / 0.034   -45.3    -42.9     50.3x (0.0731)
  DDPM, SDE, NFE 50    [491]        0.890 / 0.069 / 0.041   160.2    -51.0   1215.8x (1.7676)
  N(0, I) prior, no flow [59]       0.419 / 0.195 / 0.385    57.2    -15.3    306.1x (0.4451)
```

Gate A (energy ≤ 10× floor, every basin within 0.03 of quadrature): PASS at 3.2× / 0.029.
`[k in wall]` = samples with U > 1000 (the highest saddle is −41), counted as 1000 in ⟨U⟩. The
model over-weights A by 3 points because the training set does by 4; the reweighted row uses the
model's own log-density (`.logp.bin`) as importance weights and recovers the quadrature row.

⚠ The KL columns need the exact log-density of the DISCRETE map: with the trace form
`-h·tr J` (the continuous continuity equation) both came out slightly negative (−0.034 / −0.019);
`log|det(I + hJ)|` per Euler step fixes it (0.038 / 0.048). At NFE 1 the reflow arm's
KL(m||e) is −0.13 because a single step folds: the map is not injective and the per-path
density undercounts. The column is meaningful from NFE 2 up.

## Table 2 — temperature transfer (`transfer_table.txt`, `transfer.json`)

```
   kT  source                             p_A / p_B / p_C          <U>    dF_AB   ESS
   12  quadrature (exact)                 0.949 / 0.042 / 0.009  -131.4    -37.4
       Langevin from B, 2e5 steps         0.869 / 0.103 / 0.029  -126.2    -25.6
       flow at kT = 20, reweighted        0.960 / 0.032 / 0.008  -131.8    -40.8   1435
         ^ corrected by 1/p_theta         0.949 / 0.042 / 0.009  -131.8    -37.4   1394
    8  quadrature (exact)                 0.991 / 0.009 / 0.001  -137.9    -38.0
       Langevin from B, 2e5 steps         0.000 / 0.925 / 0.075   -96.2     --
       flow at kT = 20, reweighted        0.993 / 0.006 / 0.000  -138.0    -40.7   1001
         ^ corrected by 1/p_theta         0.991 / 0.008 / 0.001  -138.3    -38.3    958
```

Gate B (ΔF within 2 units of quadrature at kT' = 12): the naive reweighting, which assumes
p_θ = p_20, MISSES by 3.4 (it inherits the model's A-bias in full); the corrected weights
exp(−U/kT')/p_θ — Noé et al.'s — PASS at 0.0, and 0.3 at kT' = 8. The scorer gates the
corrected row when a `.logp.bin` is present. The kT = 8 chain from B made no crossing in
2×10⁵ steps (the mock's seed crossed once; this one did not), so its ΔF does not exist.
`reweighted_kT*.bin` / `corrected_kT*.bin` are the resampled clouds the figure's panel (c) draws.

## Table 3 — coupling × NFE (`table_coupling.txt`, `*_nfe_sweep.out`)

energy distance × floor; straightness and kinetic energy at NFE 50 from the exe's stdout

| coupling | NFE 1 | 2 | 5 | 10 | 50 | straightness | ∫E‖v‖²dt |
|---|---|---|---|---|---|---|---|
| independent | 233 | 91 | 20 | 8.6 | 3.2 | 0.521 | 1.71 |
| minibatch OT | 7.2 | 5.4 | 4.7 | 4.9 | 5.1 | 0.0042 | 1.26 |
| reflow | 4.9 | 4.1 | 3.7 | 3.7 | 3.7 | 0.0003 | 1.27 |

Straightness is Liu's ∫E|v(x_t,t) − (z − x)|²dt on the ODE's own (z, x) pairs, 10 t-points.
Both straightened arms plateau above the independent one at NFE 50: their training set is the
first model's output. One checkpoint per arm; no seed study.

## Field error (`field_error.txt`)

`flow reuse … field` dumps v_θ on a 64×64 lattice × 11 t; the scorer computes the exact marginal
velocity (x − E[x₀|x_t])/t by quadrature. E_pt|v_θ − v*|² runs 0.031 (t = 0.02) → 0.011 (t = 0.7)
→ 0.024 (t = 1), 0.6–3 % of E|v*|²; exact kinetic energy 1.62 averaged over t.

## Files

`samples/` holds every sample cloud (`diffusion2d_samples_muller_brown[-arm]_<sampler>_s20000_n<NFE>_e0.bin`,
f32 [2048, 2] standardised coords) with `.noise.bin` (the starting noise, per-point seeds shared
across arms), `.logp.bin` (log p_θ per sample), `.paths.bin` (51 × 128 × 2 trajectories),
`.reflogp.bin` (log p_θ on the 4096 exact points) and `.field.bin/.txt` where produced.
`boltzmann_mb.png` is the figure (`python3 scripts/boltzmann_figure.py <this dir> out.png`);
the book's copy is `blueprint/src/figures/demos/boltzmann_mb.png`.
