# diffusion_2d_demo.md — a diffusion demo you can be *wrong* about

Scoped 2026-08-26 on ares (6× RTX 4060 Ti, CUDA 12.9, PJRT 0.114).
Successor in intent to `planning/ddpm_demo.md` / `_v2` / `_v3`, which built the
image demos and hit a wall this one is designed to walk around.

⬅⬅ **START HERE, 2026-08-28: read §5.7.** Phases 0 and 5 close — all four targets
are built, trained and scored, and the reverse-process strip exists. ⛔⛔ Building
the strip found that `Ddpm.sampleNoise` **was not sampling from `N(0, I)`**: over
the 2048 seeds this demo uses, it drew two distinct radii, so every sample started
on a circle. Fixed and gated; the fix nearly halves the energy distance.
▶ **Two things are now owed.** §5.6's η sweep was run on the ring and its numbers
are superseded — whether η ≈ 0.25 still wins is untested. And phase 6's CI gate is
half done: the scorer exits non-zero and nothing calls it.

⭐ **§7's image-side proposal is half built as of the same day.** `mnist-ddpm-score`
carries this demo's metric suite onto MNIST using Chapter 3's verified CNN, and its
first run says the image DDPM checkpoint is not producing digits. See §7.

---

## 0. ⭐⭐ THE ONE-PARAGRAPH VERSION

Every other demo in this repo produces something a reader can be **wrong** about
— YOLO boxes land on faces or they do not, a UNet mask matches the trimap or it
does not. Unconditional image generation is the only one where the success
criterion is "does this look like a digit to you," and `ddpm_demo.md`'s own
results table says what that costs: **50 epochs for legible MNIST, 3M params and
7 hours for recognizable CIFAR**, with the doc's key finding being that *MSE
plateaus while sample quality keeps improving, so loss is not a proxy.* A
diffusion demo on a 2-D toy distribution inverts all of that. It trains in
**seconds on a tiny MLP**, the output is a scatter plot against a known density
so correctness is visible rather than asserted, and — the part images cannot give
you — **the whole thing has real metrics**, because you can compute a distance
between two point clouds. ▶ It is not a replacement for the image demos. It is
the one that teaches the *mechanism*, and it is the only diffusion artifact here
that can be gated in CI.

---

## 1. ⭐⭐⭐ WHY THIS AND NOT MORE DDPM — the three problems it fixes

| problem with the image demos | what 2-D does |
|---|---|
| Success is a vibe check; no metric exists | **Energy distance / MMD** against the target sample is a number, computed in ms |
| Cheapest legible result is 50 epochs (MNIST) or 7 h (CIFAR) | Seconds. Fits in a test, not a run log |
| Failure looks like success — loss drops the whole way | Mode collapse is **visible and countable** (see §4) |
| Sampler bugs are invisible under undertrained output | A broken reverse process misses the manifold *obviously* |

▶ That last row is the practical argument. `ddpm_demo_v3.md` records a **sampler
regression that had to be root-caused** (committed `408f15a`) — on image output,
a wrong sampler and an undertrained model look identical. On a spiral they do
not: the points are on the curve or they are in a blob at the origin.

---

## 2. THE TARGETS — four distributions, increasing in what they catch

Each is a 2-D density we can sample exactly, so the "ground truth" is a second
point cloud rather than a judgement.

| target | why it is in the list | the failure it catches |
|---|---|---|
| **8-gaussians** (ring of 8 modes) | the standard generative-model toy | **mode collapse** — count how many modes got hit |
| **two-moons** | two curved, interleaved manifolds | over-smoothing; the model bridging the gap between moons |
| **spiral** | one long thin manifold, high curvature | the classic: samples cut the corner instead of following the arm |
| **checkerboard** | disconnected support, sharp edges | probability mass leaking into the empty squares |

⭐ Start with **8-gaussians and spiral**. The first is countable (§4), the second
is the one that looks wrong to the eye instantly, and between them they cover
"did it find all the modes" and "did it find the manifold."

✅ **All four are built, trained and scored as of 2026-08-28** (§5.7), and each of
the three added that day caught its predicted failure on its first run:
checkerboard leaks 31.35 % of its mass into the empty squares, spiral puts
27.83 % more than 4σ off the curve, and two-moons does *not* bridge the gap
(4.88 %) — a negative result on the one target that was supposed to show
over-smoothing.

---

## 3. THE MODEL — no new primitives, and that is the point

The denoiser is an MLP: `(x, y, t) → ε̂ ∈ ℝ²`. Concretely
`.dense 2+E → 128, .relu, .dense 128 → 128, .relu, .dense 128 → 2`, with `E` the
width of the time encoding. At 128 hidden that is **~35 K parameters**, against
the tiny image UNet's 118 K.

▶ **Every layer it needs is already proved.** `.dense` is Chapter 1's,
`.relu` is Chapter 2's, and the whole net is the shape `mlpVerified` already
is — so this rides `VLayer.toSpecs` and the existing MLP render with no codegen
work at all. Compare `unet3d.md`, where the question was whether a backend would
lower rank-5 convolution.

Time conditioning reuses what `LeanMlir/Ddpm.lean` already has:

| existing | reuse here |
|---|---|
| `prependSinCosT` (sinusoidal, `2·nFreq` channels) | the same encoding, concatenated to the 2-vector instead of to an image |
| `cosineSchedule` (Nichol & Dhariwal) | unchanged |
| `ddimStep` (η = 0 affine update) | unchanged |
| `stepInputs` (per-step `t_b`, ε, `x_t`) | unchanged |
| `gaussN` (Box–Muller) | unchanged, and it is also the data sampler |

⛔ **No `prependTChannel`.** That one is image-shaped (it fills an H×W plane).
The scalar variants beside it are the right peers.

---

## 4. ⭐⭐ THE METRICS — this is the whole reason to build it

Three numbers, all cheap, all computable against a fresh sample of the true
density. This is what no image demo in the repo has.

1. **Energy distance** between generated and true clouds. Standard two-sample
   statistic, `O(n²)` on n = 2048 points, which is microseconds. Scale-free
   enough to compare across targets.
2. **Mode recall** (8-gaussians only): assign each sample to its nearest mode
   centre, count distinct modes with ≥ 1 % of mass. **8/8 or it collapsed.** An
   integer, and the single most diagnostic number in the demo.
3. **Off-manifold fraction** (spiral, checkerboard): share of samples further
   than τ from the support. Catches corner-cutting and leakage.

▶ Report all three **per sampler** (DDPM η = 1 vs DDIM η = 0) and **per step
count** (50 / 100 / 500). That table is the demo's real output, and it is the
experiment `ddpm_demo_v3.md`'s η-generalized sampler deserved and never got,
because on images there was nothing to put in the cells.

---

## 5. WHAT IT SHOWS THAT IMAGES CANNOT

⭐ **The reverse process is watchable.** Plot `x_t` at t = T … 0 as a strip of
scatter plots and the cloud visibly contracts from an isotropic Gaussian onto
the manifold. On a 28×28 image the intermediate states are grey mush; here every
frame is legible. That strip is the single best figure the diffusion half of the
book could carry, and it costs one extra dump per sampling step.

Two more, both free once the above exists:

- **The schedule is legible.** Re-run with linear vs cosine β and the difference
  shows up in the strip rather than in a loss curve.
- **DDIM vs DDPM is legible.** η = 0 is deterministic, so the same seed traces
  the same path; η = 1 does not. Side by side on 2-D, that is obvious.

---

## 5.5 ✅ BUILT 2026-08-27 — phases 0-4 and 6 land; the metrics earn their keep

Working end to end on XLA/PJRT. `lake exe diffusion-2d [train_steps] [sampler_steps]`
→ `scripts/toy2d_metrics.py`. **Trains in 5.4 s at 3,000 steps / 28 s at 20,000**
(the plan said "seconds"; it is seconds), **18,178 params**.

⚠ Correction to §3 above, and to this section's first commit (`8bc83c7`), which
reported 35,586: the measured count is **18,178**. §3's "~35 K parameters"
contradicts §3's own layer list — `10 → 128 → 128 → 2` is
`1408 + 16512 + 258 = 18,178`; reaching ~35 K needs a *third* hidden layer
(34,690). The estimate was carried into the write-up instead of the number the
binary printed, which is the same class of mistake this demo exists to catch.

| phase | state |
|---|---|
| 0 data samplers | ✅ `preprocess_toy2d.py` — **all four** targets as of 2026-08-28 (§5.7), each with an **independent** reference draw (seed 1) so the metric never scores a model against its own training set |
| 1 NetSpec | ✅ `.dense 10→128→128→2`; no new primitives, as predicted |
| 2 train | ✅ `Ddpm.stepInputs` + `prependSinCosT` at `H=W=1`, unchanged |
| 3 sample | ✅ `ddimStep`, η = 0 |
| 4 metrics | ✅ cell recall, energy distance, off-support, PPM scatter |
| 5 reverse strip | ✅ **2026-08-28** (§5.7) — `strip` flag + `scripts/toy2d_strip.py` |
| 6 CI gate | ⚠ `toy2d_metrics.py` exits non-zero on failure, but **nothing calls it**: no workflow under `.github/workflows/` and no shell runner mentions it. The exit code is written; the job is not. |

### The result, and it is the argument for this demo

| train | sampler | mode recall | off-manifold | energy (× floor) |
|---|---|---|---|---|
| 3,000 | 50 | 8/8 | 42.3% | 35.7× |
| 20,000 | 50 | 8/8 | 15.7% | 44.8× |
| 20,000 | **200** | 8/8 | **5.0%** | **30.4×** |
| 20,000 | 500 | 8/8 | 7.4% | 27.8× |

⭐⭐ **The metrics caught what a picture would not.** At 3,000/50 the scatter shows
eight clean blobs and reads as a success; the numbers say **42% of the mass is
more than 4σ off any mode**. That is the §1 claim ("failure looks like success")
reproducing on the first run of the demo built to test it.

⭐ **Plan §8, answered: 50 steps is too few, and ~200 is the knee.** Off-manifold
mass falls 15.7% → 5.0% going 50 → 200, then stops improving (7.4% at 500). The
image demos' 50 is a convention, and on this target it is measurably wrong.

▶ **The residual is not what was expected.** After 200 steps the samples are ON
the modes — the failure is *mass allocation between* them: per-mode share runs
**6.3% to 18.8%** against a true 12.5%. Mode recall is 8/8 and cannot see this;
energy distance sits at 30× the true-vs-true floor entirely because of it.

⚠ That points at η, not at capacity or steps. `ddimStep` is **η = 0, fully
deterministic**, so which mode a sample lands on is decided by how the learned
field partitions the noise plane, and an imperfect partition mis-allocates mass
permanently — extra steps cannot fix it.

## 5.6 ⭐⭐ THE η SWEEP — hypothesis REFUTED, and the answer is η ≈ 0.25

§5.5 predicted: *"η = 1 should even out the per-mode mass while leaving mode
recall at 8/8."* **Both halves are wrong.** Swept on ONE fixed checkpoint (20k
steps, `reuse` flag — training is not reproducible run-to-run, so a sweep that
retrained per point would confound η with a different model), 200 sampler steps:

| η | recall | off-manifold | energy | per-mode mass |
|---|---|---|---|---|
| 0.00 (DDIM) | 8/8 | 4.98% | 0.0244 | 6.3–18.8% |
| **0.25** | 8/8 | 6.69% | **0.0104** | **8.1–17.5%** |
| 0.50 | 8/8 | 9.91% | 0.0326 | 4.9–23.7% |
| 0.75 | 8/8 | 18.95% | 0.0850 | 1.9–37.8% |
| 1.00 (DDPM) | **7/8** | 39.40% | 0.1024 | 0.4–24.9% |

▶ η = 1 does not even out the mass — it **loses a mode entirely** (0.4%) and
*widens* the spread. But the mechanism behind the prediction is real at small η:
**η = 0.25 more than halves the energy distance** (0.0244 → 0.0104, 30× → 13×
the floor) and gives the tightest per-mode mass of any arm. A little
stochasticity repairs the deterministic partition; a lot destroys it.

⚠ **The obvious confound was checked and ruled out.** η = 1 is ancestral
sampling, designed for the full T = 1000 trajectory, so its failure at 200
strided steps could have been a step-count artifact. Re-run at 1000 steps, same
weights: recall recovers to 8/8 and off-manifold falls 39.4% → 8.4%, **but
energy gets WORSE, 0.1024 → 0.1723** (215× the floor) — the mass split hardens
into four modes at ~3% and four at 15–24%. η = 1 is intrinsically wrong here.

Best configuration measured: **η = 0.25 at 1000 steps — energy 0.0098, 12× the
floor**, the closest this demo has come to the true density.

⭐ No new primitive was needed. `ddimStep` computes `a·x + b·e`, so the σ_t term
is a second call with `(1.0, σ_t, z)`; the generalized coefficients
(Song et al. eq. 12) are computed host-side in the demo.

⚠ Still single-seed, and off-manifold and energy **disagree** about η's
direction (off-manifold prefers η = 0, energy prefers η = 0.25). With n = 1 that
disagreement is not resolved, and it is a reason to keep reporting both rather
than collapsing to one score.

### ⚠ Codegen change this required

The rank-2 DDPM loss branch (`MlirCodegen.lean`) declared `%y_ddpm` rank-2, but
`iree_ffi_train_step_adam_ddpm` hardcodes `ranks[np+1] = 4`. Callers now pass
`ddpmOutShape := [B, N, 1, 1]` and the branch reshapes — the same trick the FPN
detector already used to ride that FFI untouched. No shim change.

### ⚠ Caveats

* Single seed throughout. The energy-distance column moves non-monotonically
  (44.8× at 50 steps vs 35.7× with 7× less training) and that spread is not
  resolved at n=1.
* The 0.05 energy gate is a **regression** bar set from the observed spread, not
  a quality target. A collapsed model scores an order of magnitude worse.
* ⚠⚠ Two of those three were not even GENERATED — `preprocess_toy2d.py` wrote
  8-gaussians and spiral and nothing else, so "generated but unused" was true of
  one target and false of two. All four exist and are scored as of §5.7.

## 5.7 ⭐⭐ 2026-08-28 — PHASES 0 AND 5 CLOSE, and the strip found a bug in the noise

Evidence: `runs/2026-08-28-toy2d-strip-and-targets/`. Three things, and the third is the one a
reader should not have to rediscover.

### The four targets exist and each catches its own failure

⚠ §5.6's caveat said spiral, two-moons and checkerboard were "generated but unused". One of those
three was generated. The other two were not written at all, so phase 0's "four targets" had been
half done and recorded as done. All four now ship with an independent reference draw, a support
cloud and **equal-mass cells**, which is what lets one 1 %-of-mass recall threshold mean the same
thing on every target.

20k train steps, 200 sampler steps, η = 0.25, post-fix:

| target | cells | recall | energy | × floor | off-support | per-cell mass |
|---|---|---|---|---|---|---|
| 8-gaussians | 8 modes | 8/8 | 0.00545 | 7× | 8.59 % (τ = 0.20) | 9.8–17.1 % |
| spiral | 8 arcs | 8/8 | 0.01161 | 23× | **27.83 %** (τ = 0.04) | 7.9–19.8 % |
| two-moons | 2 moons | 2/2 | 0.00580 | 3× | 4.88 % (τ = 0.20) | 48.0–52.0 % |
| checkerboard | 8 squares | 8/8 | 0.00526 | 4× | **31.35 %** (exact) | 6.3–10.6 % |

⭐⭐ **Every new target earned its place on its first run**, which §2 predicted and nothing had
tested. Checkerboard leaks **31.35 %** of its mass into squares the density provably does not
occupy — a closed-form test, not a τ estimate, and the only exact off-support number in the demo.
Spiral cuts corners at **27.83 %**. Two-moons is the clean one at 4.88 %, so the bridging failure
it was chosen for does not occur here; that is a negative result and worth keeping as one.

⚠⚠ **The phase-6 gate cannot see any of that.** It is recall plus energy distance, and
checkerboard passes both while putting a third of its mass in empty space. A demo whose argument
is "failure is countable here" has a gate that does not count this one. ▶ Adding an off-support
bound is the obvious fix and it needs a per-target baseline nobody has measured.

⭐ The scorer generalised from "mode recall against 8 hardcoded centres" to "cell recall against a
support cloud", and reproduces §5.6's committed η = 0.25 row **exactly** on the pre-fix samples —
0.01042, 6.69 %, 8.1–17.5 %. The generalisation was checked against the old numbers before it was
trusted with new ones.

### ⛔⛔ `Ddpm.sampleNoise` was not sampling from `N(0, I)`

The first panel of the reverse-process strip is meant to be an isotropic blob. It was a **ring**.

`lean_ddpm_sample_noise` seeded its xorshift64 by XOR alone and read the first uniform from the
**top** 53 bits. xorshift64 is linear over GF(2), so seeds differing in low bits still differ only
in low bits after one round: the top of the word never moves, `u1` never changes, and the
Box–Muller radius `√(-2 ln u1)` is constant. Over the 2048 seeds the demo uses for `x_T` there
were **two distinct radii**, both 1.9130.

⭐ **Why nothing caught it.** Per-axis mean and variance are correct under the defect — the mass is
on a shell rather than filling the ball — so every summary statistic over the coordinates agrees
with a healthy Gaussian. Only the radius separates them. And the image DDPMs draw one long vector
per call, where only the first pair is correlated across seeds; the 2-D demo draws `n = 2` per
point and so saw nothing else.

Fixed with SplitMix64 seeding and gated by `lake exe test-sample-noise-seeding`, which asserts
`|x|` is Rayleigh over the three seed patterns the driver actually uses. Hermetic, no GPU.

**Same checkpoint, same sampler, seeding the only difference:**

| | energy | × floor | off-support | per-mode mass |
|---|---|---|---|---|
| `x_T` on a circle | 0.01042 | 13× | 6.69 % | 8.1–17.5 % |
| `x_T ~ N(0, I)` | **0.00545** | **7×** | 8.59 % | 9.8–17.1 % |

⭐ The fix nearly halves the energy distance and beats the best figure this demo had ever recorded
(0.0098 at 1000 sampler steps, §5.6). Feeding the reverse process a distribution it was never
trained on cost real quality.

⚠⚠ **§5.6's η sweep is superseded in its NUMBERS.** Every row of it was sampled from the ring.
Whether its conclusion — η ≈ 0.25 beats both η = 0 and η = 1 — survives is **not tested**, and
re-running that sweep on the fixed noise is the first thing owed here.

### The strip, and the axis it is spaced on

Frames are spaced uniformly in **log σ**, where `σ_t = √(1-ᾱ_t)` is the noise scale of the
marginal `p(x_t) = data ⊛ N(0, σ_t²)`. Both obvious alternatives were computed and rejected:
uniform in `t` puts **five of nine panels above σ = 0.7** — measured by running the `tframes`
branch, not argued — and uniform in `ᾱ` is worse still, because the cosine schedule moves ᾱ
fastest exactly where nothing is visible yet. Log σ
lands at σ = 1.00, 0.58, 0.34, 0.20, 0.11, 0.06, 0.04, 0.02, which brackets the 8-gaussians' own
0.05 mode width, so the modes appear **across** panels rather than between the last two.
`tframes` restores the naive spacing so the difference can be seen rather than taken on trust.

▶ **Both findings in this section came from looking at the figure**, not from a metric. Nothing in
§4's suite moved when the noise was wrong, and nothing in it says which axis to space panels on.
That is the §5 claim — "the reverse process is watchable" — paying for itself twice before the
figure was even finished.

---

## 5.8 ⭐⭐ 2026-08-28 — THE SCORE-SDE SAMPLERS, and two predictions refuted

Evidence: `runs/2026-08-28-toy2d-strip-and-targets/sampler_nfe_sweep.log`,
`scripts/toy2d_sampler_sweep.sh`.

⭐ **No retraining was needed.** Under the VP SDE the score is `-ε̂(x,t)/σ_t`, so the ε-predicting
network this repo already trains *is* a score model up to that factor. Everything here is a
sampler: `abarC`/`sigC`/`betaC` give the cosine schedule as a continuous function (β in closed
form, not differenced), and three solvers ride the existing `ddimStep` and `F32.axpySlice`.
**No new codegen, no new FFI, no new layer.**

Energy distance on 8-gaussians, 512 samples, ONE checkpoint held fixed across every arm, x-axis in
network evaluations rather than steps (Heun spends two per step, so at NFE 20 it takes 10):

| NFE | ddim η=0 | ddim η=0.25 | euler | heun | sde |
|---|---|---|---|---|---|
| 10 | 0.0560 | 0.0479 | 1.2504 | 0.8120 | **0.0294** |
| 20 | 0.0278 | 0.0262 | 0.4045 | 0.2931 | **0.0112** |
| 50 | 0.0116 | 0.0093 | 0.1125 | 0.1049 | **0.0031** |
| 100 | 0.0087 | 0.0076 | 0.0560 | 0.0515 | **0.0034** |
| 200 | 0.0081 | 0.0069 | 0.0323 | 0.0287 | **0.0035** |

⛔ **REFUTED #1: "Heun wins at low NFE."** It loses, at 14.5× DDIM's error at NFE 10 and still
3.6× at 200. Second-order accuracy does not rescue a stiff problem — the linear part has to be
handled semi-analytically, which is what DDIM's exponential form does and what an explicit method,
however high its order, does not. ▶ The gap between DDIM and naive Euler at NFE 10 is **22×**.

⭐⭐ **The reverse SDE wins at every budget and saturates early.** At NFE 50 it reaches 0.0031 —
better than DDIM manages at ANY budget here — and 4× more compute buys nothing (0.0034, 0.0035).
Injected noise correcting accumulated score error is the standard explanation, and on a small
imperfect denoiser that correction is worth a lot.

✅ **§5.6's η finding holds across the whole range**, not just at 200 steps: η = 0.25 beats η = 0 at
every budget by ~15 %.

### ✅ The spacing confound, settled — and it was most of the effect

Two further sweeps, same weights, same NFE grid, only the time grid changing.
`sampler_spacing_logsnr.log`, `sampler_spacing_logabar.log`.

⛔ **log-σ spacing is catastrophic for every solver** — Euler 1.25 → **25.1** at NFE 10, and even
the SDE 0.029 → **31.2**, with essentially all mass off-support. It concentrates steps at small σ
and takes one enormous step across exactly the region where β diverges. ▶ Worth keeping in view:
log σ is the RIGHT axis for the §5.7 strip, because it is where the picture changes, and the WRONG
axis for the solver, because that is not where the stiffness is. The same schedule wants different
parameterisations for different jobs.

⭐ **log-ᾱ spacing is the stability-optimal grid** (β = -d/dt log ᾱ, so `h·β` — the amplification
factor in the Euler update — is held constant), and on it Euler is a different sampler:

| NFE | euler, uniform-t | euler, **log-ᾱ** | ddim η=0 |
|---|---|---|---|
| 10 | 1.2504 | **0.0874** | 0.0560 |
| 20 | 0.4045 | **0.0447** | 0.0278 |
| 50 | 0.1125 | **0.0341** | 0.0116 |
| 100 | 0.0560 | **0.0283** | 0.0087 |
| 200 | 0.0323 | **0.0217** | 0.0081 |

⛔⛔ **REFUTED #2, and it is my own claim from an hour earlier.** "Naive Euler is 22× worse than
DDIM" was mostly a SPACING artifact. Given its own best grid Euler improves **14×** and comes
within 1.6× of DDIM at NFE 10.

⭐ **The residual is real and it is the integrator.** The gap does not close with compute — it
*widens*, 1.6× at NFE 10 to 3.3× at NFE 100 — because DDIM keeps converging while Euler plateaus
(0.087 → 0.022 for 20× the compute). A convergence-rate difference at matched spacing is an
integrator property, not a parameterisation one. ▶ So the exponential-form sentence can be written,
but it is worth ~3×, not ~20×.

⚠⚠ **And there is no single best grid.** The SDE gets WORSE on log-ᾱ (0.765 at NFE 10 against
0.029 on uniform-t), so the explicit ODE solvers and the SDE want opposite grids. The first sweep's
design — one grid for every arm — was wrong in both directions at once, and the fair comparison is
each solver on its own best grid. On that basis the ranking at NFE 50 is **sde/uniform-t 0.0031 <
ddim η=0.25 0.0093 < euler/log-ᾱ 0.0341**.

⚠ **Cell recall cannot see any of this** — every arm scores 8/8, so at this model quality the
sharpest metric of §4 is saturated and the energy distance is carrying the whole comparison.

⚠ **And the two metrics disagree most sharply here yet**: the SDE at NFE 10 has the BEST energy
distance of any arm at that budget while sitting **76 % off-support**. §5.6 recorded that
disagreement; this is its extreme case.

⚠ Single seed, 512 samples per arm, one target, one checkpoint. The t-grid is also quantised to the
1000 training indices (the encoder takes an integer), which is negligible at these budgets but
would bite below NFE ≈ 10.

### ⛔⛔ 5.8b — THE TOY'S SAMPLER RANKING DOES NOT SURVIVE AT IMAGE SCALE

`runs/2026-08-28-mnist-ddpm-verified-score/sampler_sweep_mnist.log`,
`scripts/mnist_ddpm_sampler_sweep.sh`. The same three solvers, ported through the shared `Ddpm`
schedule, against the 50-epoch MNIST model, scored by Chapter 3's verified CNN — a different
target, a different denoiser, a different metric family.

| sampler | NFE 10 | NFE 20 | NFE 50 | NFE 200 |
|---|---|---|---|---|
| **ddim** | **0.0874** | **0.0437** | **0.0237** | **0.0188** |
| sde | 1.1837 | 0.9870 | 0.3382 | 0.0211 |
| euler | 0.3542 | 0.2053 | 0.0910 | 0.0288 |

⛔⛔ **The toy said the reverse SDE wins at every budget and saturates by NFE 50. On MNIST it
DIVERGES** — pixel mean 2.90 against the data's 0.13, coverage 2/10 — and needs NFE 200 to come
back to within 12 % of DDIM. The ranking inverts completely. ▶ **This is the toy doing its job.**
"Overfit on the toy, then scale" is supposed to catch a finding that does not transfer, and this is
one; the cost of learning it was ten minutes rather than a chapter written around a wrong claim.

⚠ **What is settled and what is not.** Settled: *this Euler-Maruyama discretisation* does not
transfer — stable in 2-D with a well-fitted denoiser, unstable in 784-D. NOT settled: stochastic
versus deterministic sampling in general, because the stable discrete form of the reverse SDE is
ancestral sampling (DDIM at η = 1) and the MNIST driver hardcodes η = 0. ▶ Porting the η knob is
the missing arm.
⚠ The two problems differ in dimension, denoiser (MLP vs UNet), training quality and data all at
once, so "dimension is why" is not established either.

⭐ **Euler's residual gap is consistent across both problems.** On MNIST it is 1.5× DDIM at NFE 200;
on the toy, once given its own best grid, 2.7×. Same order, two very different problems — which is
the best evidence yet that the integrator difference is real and the ~20× from the first sweep was
spacing.

⭐⭐ **And confidence lies for the third time today.** The DIVERGED sampler at NFE 10 posts
**99.08 %** confidence — the highest number in the table — while covering 2 of 10 classes with
pixels an order of magnitude out of range. A single quality score would rank it first.

### ⛔⛔ 5.8c — ANCESTRAL SAMPLING (η = 1) WINS ON MNIST, and §5.8b's reading was wrong

`runs/2026-08-28-mnist-ddpm-verified-score/eta_sweep_mnist.log`. The missing arm, run: DDIM's η
knob on the MNIST driver, so the stable discrete form of the reverse SDE could be measured beside
the Euler–Maruyama one that diverged.

Energy distance against real MNIST, 1024 samples, same 50-epoch checkpoint:

| NFE | η = 0 | η = 0.25 | η = 0.5 | **η = 1 (ancestral)** |
|---|---|---|---|---|
| 10 | 0.0874 | 0.0822 | 0.0713 | **0.0455** |
| 20 | 0.0437 | 0.0383 | 0.0275 | **0.0132** |
| 50 | 0.0237 | 0.0172 | 0.0137 | **0.0067** |
| 200 | 0.0188 | 0.0134 | 0.0130 | **0.0099** |

⭐⭐ **Monotone in η at every budget, and ancestral is best by 2–3×.** η = 1 at NFE **50** scores
0.0067 — nearly 3× better than deterministic DDIM at NFE **200** (0.0188) on a quarter of the
compute.

⛔⛔ **So §5.8b's Euler–Maruyama divergence was ENTIRELY a discretisation artifact.** Stochastic
sampling is not worse on MNIST; it is substantially better. The hedge written into the book
("a statement about this discretisation rather than about stochastic sampling") was the right call
and is now confirmed rather than merely cautious.
▶ **`blueprint/src/content.tex` §10.2.7 is now misleading and needs an edit.** Its solver table
shows η = 0 DDIM, Euler–Maruyama and PF-ODE Euler, and concludes "DDIM wins at every budget". True
of those three arms, false of the family: the best sampler measured is ancestral.

⛔ **And the toy mis-ranked η too.** §5.6 found η = 0.25 best and η = 1 losing a mode. On MNIST
η = 1 is the winner outright and η = 0.25 is third of four. That is the second toy conclusion to
fail transfer, after §5.8b's.
⚠ Single seed, one checkpoint, one target. η = 1 is also non-monotone in NFE (0.0067 at 50 against
0.0099 at 200), unexplained and not chased.

---

## 6. PHASES

| # | what | est. | gate | state |
|---|---|---|---|---|
| 0 | Data samplers for the four targets (host-side, `gaussN` + a few lines each) | S | a scatter of the *true* density looks right | ✅ 08-28 |
| 1 | `NetSpec` for the MLP denoiser + wire `prependSinCosT`'s scalar peer | S | shapes tie to `toSpecs`; render compiles | ✅ 08-27 |
| 2 | Train loop; reuse `stepInputs` unchanged | M | loss drops; **seconds per run** | ✅ 08-27 |
| 3 | Sample with the existing `ddimStep`; PPM/PNG scatter out | S | 8-gaussians visibly hits 8 modes | ✅ 08-27 |
| 4 | ⭐ The three metrics of §4 + the sampler × steps table | M | 8/8 mode recall, energy distance reported | ✅ 08-27, generalised to four targets 08-28 |
| 5 | The reverse-process strip (§5) | S | the figure | ✅ 08-28 (§5.7) |
| 6 | ⭐⭐ CI gate: mode recall = 8/8 and energy distance < threshold | S | runs in the corpus job, seconds | ⚠ half |

▶ **Phase 6 is the payoff and should not be dropped.** It makes this the first
generative model in the repo with a regression test. Every image demo is
un-gateable by construction; this one is a handful of seconds and an integer.

⚠⚠ **Phase 6 is HALF done and was recorded as whole.** `toy2d_metrics.py` exits
non-zero on failure, which is the easy half. Nothing calls it: no workflow under
`.github/workflows/` and no shell runner in the tree mentions it, so the "runs in
the corpus job" in its own gate column has never been true. The exit code is
written; the job is not. ▶ That is the cheapest thing outstanding on this demo.

⚠ And when it is wired, note what it will and will not catch — §5.7 measures a
model passing both halves of the gate while leaving 31 % of its mass in provably
empty space.

---

## 7. ⚠ SCOPE — what this deliberately is not

- ⛔ **Not a replacement for the image demos.** It exercises none of the conv
  kit, no BatchNorm, no attention. It proves the diffusion *math* and the
  sampler, not the vision stack. Both should exist.
- ⛔ **Not a research artifact.** 2-D toys are a debugging instrument. No claim
  about them transfers to images, and the doc should say so where a reader might
  hope otherwise.
- ⛔ **Not flow matching.** Better samples per FLOP and simpler paths, but it is
  a different algorithm and the bestiary catalogues DDPM/Stable Diffusion. One
  thing at a time; if the 2-D harness exists, adding rectified flow beside it
  later is cheap *because* the metrics of §4 already exist to compare them.
- ✅⭐ **The UNCONDITIONAL half of this is BUILT, 2026-08-28** — `mnist-ddpm-score`
  + `scripts/mnist_ddpm_score.py`, evidence in
  `runs/2026-08-28-mnist-ddpm-verified-score/`. Push the samples through
  Chapter-3's `cnnVerified` and every statistic of §4 comes back in the space it
  maps images into: class coverage (the 8/8 mode-recall analogue), per-class
  mass, confidence, and this doc's own energy distance over the 10-d softmax.
  **The metric is bracketed at both ends** — real MNIST scored as if generated
  sits at the floor, unstructured pixels at 231× it.
  ⛔⛔ **And its first run said the MNIST DDPM checkpoint did not produce
  digits**: 119× the floor, 9/10 coverage, and pixels over [-4.9, 11.9] where the
  data is [0, 1]. Two causes already in the tree — `MainMnistDdpmTrain` never
  centred to [-1, 1] (its CIFAR siblings all do, and its own comment said it
  didn't) and the default was 3 epochs against `ddpm_demo.md`'s own "50 for
  legible MNIST".
  ⭐⭐ **BOTH FIXED THE SAME DAY, and the metric measured the fix.** Centring plus
  50 epochs: **15× the floor, 10/10 coverage, 90.94 % confidence, legible
  digits.** The full ladder, noise to real: **231× → 119× → 33× → 15× → 1×**.
  ⚠⚠ **The full 2×2 was run and EPOCHS dominate, not centring.** At fixed
  centring, 3 → 50 epochs moves 119× → 33× (raw) and 73× → 15× (centred); at
  fixed epochs, raw → centred moves 119× → 73× and 33× → 15×. ▶ The first
  write-up had only the 3-epoch arms and read as though centring was the fix; a
  50-epoch UNCENTRED figure from July (`demos/figures/ddpm_mnist.png`) showing
  legible digits is what forced the missing arm. **Two arms of a 2×2 do not
  settle which factor mattered.**
  ⭐⭐ **And confidence ranks the two 50-epoch arms BACKWARDS**: uncentred scores
  92.82 % against centred's 90.94 % while dropping a class (9/10, "1" at 0.8 %)
  and more than doubling the energy distance. That is §5.7's checkerboard
  argument reproducing on images — a one-number quality score passes the model a
  coverage term plus a distribution term catches.
  ⚠ Coverage moved non-monotonically across the ladder (9/10 → 8/10 → 9/10 →
  10/10). Coverage and per-sample quality are not the same axis.
  ⚠⚠ The old and new checkpoints are **shape-identical and semantically
  incompatible**, so the spec name now carries `centered` and `buildPrefix`
  keeps them apart. Nothing else would have caught a double-transform.
  ⚠⚠ **The negative control earned its place immediately.** The classifier is
  58 % confident on pure noise, so confidence spans 58–99 rather than 0–100; and
  noise puts 55 % of its mass on "5" where the DDPM puts 41.8 %, so most of the
  model's apparent obsession with fives is the classifier's own off-distribution
  prior. ▶ A per-class mass table is not interpretable without a noise arm
  beside it.
- ⚠ **The CONDITIONAL half is still a proposal.** Condition on the digit and
  accuracy becomes the headline number. It needs a class embedding in the
  denoiser; the cheapest route rides `prependSinCosT`'s own trick, since that
  already prepends channels to the image, so a one-hot is more channels and a
  wider first conv rather than the FiLM block `ddpm_demo_v3.md` workstream E
  specs.

---

## 8. THE OPEN QUESTION

How many diffusion steps does a 2-D manifold actually need? The image demos use
50 DDIM steps because that is the convention, not because anything measured it.
With §4's metrics, the step-count sweep is a table rather than an opinion — and
if the answer on a spiral is "12," that is worth knowing before defending 50 on
an image where nobody can tell.
