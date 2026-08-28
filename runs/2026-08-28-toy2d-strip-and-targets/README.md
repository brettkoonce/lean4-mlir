# 2026-08-28 — the 2-D diffusion demo gets its four targets and its reverse-process strip

Two phases of `planning/diffusion_2d_demo.md` close, and the second of them found a bug in
`Ddpm.sampleNoise` that nothing else in the repo could see.

1. **Phase 0 is finished.** `preprocess_toy2d.py` wrote two of the plan's four targets and the
   plan's own caveat said the other two were "generated but unused" — they were not generated at
   all. All four exist now, each with an independent reference draw, a support cloud and
   equal-mass cells, and all four are trained and scored.
2. **Phase 5 is built.** `lake exe diffusion-2d <target> strip` dumps the cloud at nine points
   along the reverse process and `scripts/toy2d_strip.py` renders the panel strip.
3. ⛔⛔ **`Ddpm.sampleNoise` was not sampling from `N(0, I)`.** Over the 2048 seeds the demo uses
   for `x_T` it produced **two distinct radii** — every 2-D sample started on a circle of radius
   1.913. Fixed, gated, and the fix nearly halves the energy distance.

⚠ Phase 6's CI gate is still not in CI. The scorer exits non-zero, nothing calls it.

---

## 1. The bug the strip found

The first panel of a reverse-process strip is supposed to be an isotropic blob. It was a ring.

`lean_ddpm_sample_noise` seeded its xorshift64 stream by XOR alone (`s = seed ^ K`) and then read
its first uniform from the **top** 53 bits (`s >> 11`). xorshift64 is linear over GF(2), so two
seeds differing only in low bits still differ only in low bits after one round — the top of the
word never moves. Every call therefore drew the same `u1`, hence the same Box–Muller radius
`√(-2 ln u1)`.

⭐ **Why nothing caught it.** Per-axis mean and variance are both correct under the defect: the
mass is on a shell instead of filling the ball, so every summary statistic over the coordinates
agrees with a healthy Gaussian. The radius is the only thing that separates them. The image DDPMs
draw one long vector per call (`sampleNoise (B * nPix) 0xc0ffee`), where only the first pair is
correlated across seeds and the rest comes from a well-mixed stream, so their grids looked normal.
The 2-D demo draws `n = 2` per point and so saw nothing else.

Fixed by running the seed through SplitMix64 before the stream starts — the canonical seeding for
xorshift-family generators. Gated by `lake exe test-sample-noise-seeding`, which asserts `|x|` is
Rayleigh (mean 1.25, sd 0.66, `E|x|² = 2`) over the three seed patterns the driver actually uses.
The defect scored sd ≈ 1e-6 and `E|x|² = 3.66`. Hermetic, no GPU. `seeding_gate.log`.

### The A/B, same checkpoint, same sampler settings

`noise_fix_ab.log`. 8-gaussians, 20k train steps, 200 sampler steps, η = 0.25. The only thing that
differs between the arms is the seeding.

| | recall | energy | × floor | off-support | per-mode mass |
|---|---|---|---|---|---|
| before, `x_T` on a circle | 8/8 | 0.01042 | 13× | 6.69 % | 8.1–17.5 % |
| after, `x_T ~ N(0, I)` | 8/8 | **0.00545** | **7×** | 8.59 % | 9.8–17.1 % |

⭐ The fix roughly halves the energy distance and beats the best number the demo had ever recorded
(0.0098 at 1000 sampler steps). Feeding the reverse process a distribution it was not trained on
cost real sample quality.

⚠⚠ **This supersedes §5.6's η sweep in absolute terms.** Every row of that table was sampled from
the ring. Whether its *conclusion* — η ≈ 0.25 beats both η = 0 and η = 1 — survives the fix is not
tested here and is the first thing a next session should re-run.

⚠ Off-support went slightly UP while energy went down, which is the disagreement §5.6 already
records between those two metrics. It is not resolved by this.

---

## 2. The four targets

`metrics_<target>.log`, `train_three_targets.log`. 20k train steps, 200 sampler steps, η = 0.25,
2048 samples against an 8192-point independent reference.

| target | cells | recall | energy | × floor | off-support | per-cell mass |
|---|---|---|---|---|---|---|
| 8-gaussians | 8 modes | 8/8 | 0.00545 | 7× | 8.59 % (τ = 0.20) | 9.8–17.1 % |
| spiral | 8 arcs | 8/8 | 0.01161 | 23× | **27.83 %** (τ = 0.04) | 7.9–19.8 % |
| two-moons | 2 moons | 2/2 | 0.00580 | 3× | 4.88 % (τ = 0.20) | 48.0–52.0 % |
| checkerboard | 8 squares | 8/8 | 0.00526 | 4× | **31.35 %** (exact) | 6.3–10.6 % |

⭐⭐ **Each of the three new targets caught its own failure on its first run**, which is the whole
argument for having four rather than one.

* **checkerboard** leaks **31.35 %** of its mass into squares the density provably does not
  occupy. That is not a τ estimate — square membership is a closed-form test — and it is why this
  target is in the plan's list. Its per-cell row sums to 68.6 % for the same reason.
* **spiral** puts **27.83 %** of its mass more than 4σ from the curve: the corner-cutting the plan
  predicted, on the target chosen to expose it.
* **two-moons** is the *clean* one — 4.88 % off-support, mass split 48/52 — so the bridging failure
  it was meant to catch does not occur here. A negative result, and it is the target with the
  fewest cells and the lowest energy multiple.

⚠⚠ **The gate does not look at off-support, and checkerboard shows what that costs.** A model
putting a third of its mass in empty space passes on 8/8 recall and a 0.0053 energy distance. Both
halves of the gate are insensitive to it. The demo exists to make failure countable and this is a
failure the current gate cannot count.

### The scorer was validated against truth first

`scorer_selfcheck.log` — the train draw scored against the reference draw, per target. All four
give K/K recall, equal cell mass, ~0 % off-support and an energy distance at or below the
true-vs-true floor. ⚠ 8-gaussians and checkerboard print **identical** per-cell mass rows there;
both open with `rng.integers(0, 8, n)` on seed 0, so it is the same first draw from the same seed
rather than the same data.

### And the rewrite is behaviour-preserving on 8-gaussians

The scorer generalised from "mode recall against 8 hardcoded centres" to "cell recall against a
support cloud", and it reproduces §5.6's committed η = 0.25 row exactly on the pre-fix samples:
energy 0.01042, off-manifold 6.69 %, per-mode mass 8.1–17.5 %. The generalisation was checked
before it was trusted.

---

## 3. The strip, and the axis it is spaced on

⚠ **Frames are spaced uniformly in log σ, where `σ_t = √(1-ᾱ_t)`**, and the two obvious
alternatives were rejected on numbers rather than taste. Provenance differs by row and is worth
stating: rows 1 and 3 were **run** (row 1 through the `tframes` flag, which exists so that
comparison stays reproducible), row 2 was **computed** from the cosine schedule and never shipped
as a code path.

| spacing | σ at the eight frames | what it costs |
|---|---|---|
| uniform in `t` | 1.00, 0.98, 0.92, 0.83, 0.71, 0.56, 0.39, 0.20 | **five of nine panels sit above σ = 0.7** — barely-touched noise |
| uniform in `ᾱ` | 1.00, 0.93, 0.87, 0.79, 0.71, 0.61, 0.49, 0.35 | worse — ᾱ moves fastest where nothing is visible yet |
| **uniform in log σ** | 1.00, 0.58, 0.34, 0.20, 0.11, 0.06, 0.04, 0.02 | the modes appear ACROSS panels |

The last row brackets the 8-gaussians' own 0.05 mode width, which is why it works: σ is the scale
of the blur in `p(x_t) = data ⊛ N(0, σ_t²)`, so spacing by it is spacing by how much of the
picture is left. Uniform log σ is uniform log-SNR once ᾱ ≈ 1, which is the axis the diffusion
literature plots against. `tframes` restores the naive spacing so the difference can be seen
rather than taken on trust — and it was run, which is where the top row's numbers come from
(`strip_frames.log` carries the default; the `tframes` σ row is in this table).

▶ This was found the same way the noise bug was: by looking at the figure. The first render used
uniform `t`, and the spiral's cloud was still shapeless at t = 124 of 999 and fully resolved at
t = 0.

### The figures

`strip_<target>.png`, nine panels each, left to right. The `spread` row of
`strip_frames.log` is the same story as a number — std of `|x|` across the cloud, contracting
monotonically:

| target | σ = 1.00 → 0.00 | reads as |
|---|---|---|
| 8-gaussians | 0.66 → 0.07 | blob, ring, eight modes sharpening |
| two-moons | 0.66 → 0.26 | the cleanest of the four; two interleaved arcs by σ = 0.11 |
| spiral | 0.66 → 0.23 | the arm appears, then stays visibly thick — the 27.83 % off-support |
| checkerboard | 0.66 → 0.28 | squares emerge with soft edges — the 31.35 % leak |

⭐ In all four the picture and the metric agree, which is the point: the strip is not decoration
next to the numbers, it shows the same failure the numbers count.

---

## 4. Reproducing

```
python3 preprocess_toy2d.py 8192 data/toy2d
lake exe diffusion-2d <target> 20000 200 25 strip     # eight_gaussians|spiral|two_moons|checkerboard
python3 scripts/toy2d_metrics.py --target=<target>
python3 scripts/toy2d_strip.py <target>
lake exe test-sample-noise-seeding                    # hermetic, no GPU
```

Costs on one 4060 Ti: 28 s to train, ~3.5 min to sample 2048 points at 200 steps one at a time.
⚠ `reuse` holds the weights fixed; training is not reproducible run to run, so any sweep must use
it or it confounds the swept variable with a different model.
