# 2026-10-27 — Boltzmann generator, the flow arm at four seeds (planning/part2_ship_audit.md §2.6)

The 09-11 section rests on one checkpoint (`runs/2026-09-11-boltzmann-generator/README.md:81`'s
caveat). Three more trainings of the `flow` arm — independent coupling, 20,000 steps, Adam 1e-3,
batch 256, 18,178 parameters, 33 s each on one 4060 Ti under XLA at `LEAN_MLIR_MEM_FRACTION=0.1`
— under `LEAN_MLIR_SEED=1..3`, the knob added to `demos/MainDiffusion2d.lean` the same afternoon
(it offsets the He init and the sampling noise; 0 = the 09-11 run), each sampled at NFE 50 with
`logp` and `nll` as the 09-11 row was, and scored in one table beside the 09-11 cloud (`flow_s0`,
copied from `…-boltzmann-generator/samples/`). The arm's checkpoint, which `reuse` and the
figure read, was put back after the three trainings (`run.sh`).

```
LEAN_MLIR_SEED=$s LD_LIBRARY_PATH=ffi CUDA_VISIBLE_DEVICES=0 LEAN_MLIR_MEM_FRACTION=0.1 \
  .lake/build/bin/diffusion-2d muller_brown flow 20000 50 fm-euler logp nll
python3 scripts/demos/boltzmann_metrics.py score flow_s0=flow_s0.bin flow_s1=flow_s1.bin \
  flow_s2=flow_s2.bin flow_s3=flow_s3.bin --gate --out=.          # table1.{txt,json}
python3 scripts/demos/boltzmann_metrics.py score flow_s$k=flow_s$k.bin --gate   # gates.txt, per seed
```

## Table 1 at kT = 20, n = 2048 (`table1.txt`)

```
  source                       p_A / p_B / p_C             <U>    dF_AB   energy (x floor)     KL(m||e) / KL(e||m)
  quadrature (exact)           0.806 / 0.129 / 0.065    -113.6    -36.6      1.0x (0.0015)
  Langevin, 8 chains x 1e5     0.844 / 0.101 / 0.054    -112.6    -42.4      4.1x (0.0060)
  flow_s0 (09-11)              0.834 / 0.103 / 0.063    -112.4    -41.9      3.2x (0.0046)    0.038 /  0.048
    ^ reweighted by p_20/p_theta 0.805 / 0.129 / 0.066  -112.8    -36.7                       ESS 1864
  flow_s1                      0.843 / 0.095 / 0.062    -114.9    -43.6      5.3x (0.0078)    0.036 /  0.033
    ^ reweighted               0.810 / 0.125 / 0.065    -113.0    -37.3                       ESS 1895
  flow_s2                      0.832 / 0.099 / 0.070    -113.3    -42.6      3.6x (0.0052)    0.023 /  0.038
    ^ reweighted               0.801 / 0.129 / 0.071    -112.6    -36.6                       ESS 1866
  flow_s3                      0.838 / 0.090 / 0.071    -110.6    -44.6      6.3x (0.0091)    0.076 /  0.069
    ^ reweighted               0.813 / 0.121 / 0.066    -113.1    -38.1                       ESS 1740
  N(0, I) prior, no flow       0.419 / 0.195 / 0.385      57.2    -15.3    306.3x (0.4451)
```

## Gate A, per seed (`gates.txt`)

| seed | energy distance | max basin error, raw cloud | gate A as written | max basin error, reweighted |
|---|---|---|---|---|
| 0 (09-11) | 3.2× floor | 0.029 | PASS | 0.001 |
| 1 | 5.3× | 0.038 | **FAIL** | 0.004 |
| 2 | 3.6× | 0.031 | **FAIL** | 0.006 |
| 3 | 6.3× | 0.039 | **FAIL** | 0.008 |

**The energy half of the gate holds on every seed (3.2–6.3× against the 10× bar); the basin half
fails on three of four, and the 09-11 seed passed it by 0.001.** The reason is in the Langevin
row: the training set itself puts 0.844 on basin A against the quadrature's 0.806, an error of
0.038 that would fail the same bar, and every seed's raw cloud lands between the data and the
truth (0.832–0.843). A 0.03 bar on the raw samples is a bar on the training set's bias, which the
model reproduces to ±0.006 on every seed — the single-checkpoint number was a pass at the edge of
the noise, not a property of the recipe. The number that is a property of the recipe is the
importance-reweighted row (the model's own `logp` as the weight, Noé et al.'s correction, which
the 09-11 README already used to recover the quadrature row): within 0.008 of the exact basin
weights on all four seeds, ESS 1740–1895 of 2048, and ΔF_AB within 1.5 of the exact −36.6
(−36.7 / −37.3 / −36.6 / −38.1). The KL columns sit at 0.02–0.08 nats (seed 3 the widest).

What this says for the section: the sampler-quality claim should be made on the reweighted row
with a four-seed spread, and gate A's basin criterion restated on that row (or at 0.04 on the raw
one, which is where the data's own bias sits). Nothing in Table 2 (temperature transfer) or
Table 3 (coupling × NFE) is re-run here; both read the 09-11 checkpoint, which is back in place.
Files: `flow_s{1,2,3}.log` (training), `flow_s{1,2,3}_params.bin` (the three checkpoints, 73 KB
each), `flow_s{0..3}.{bin,logp.bin,reflogp.bin}` (+ `noise.bin` for 1–3), `table1.{txt,json}`,
`gates.txt`.
