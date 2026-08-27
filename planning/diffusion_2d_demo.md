# diffusion_2d_demo.md — a diffusion demo you can be *wrong* about

Scoped 2026-08-26 on ares (6× RTX 4060 Ti, CUDA 12.9, PJRT 0.114).
Successor in intent to `planning/ddpm_demo.md` / `_v2` / `_v3`, which built the
image demos and hit a wall this one is designed to walk around.

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
| 0 data samplers | ✅ `preprocess_toy2d.py` — 8-gaussians + spiral, plus an **independent** reference draw (seed 1) so the metric never scores a model against its own training set |
| 1 NetSpec | ✅ `.dense 10→128→128→2`; no new primitives, as predicted |
| 2 train | ✅ `Ddpm.stepInputs` + `prependSinCosT` at `H=W=1`, unchanged |
| 3 sample | ✅ `ddimStep`, η = 0 |
| 4 metrics | ✅ mode recall, energy distance, off-manifold, PPM scatter |
| 5 reverse strip | ⛔ not built |
| 6 CI gate | ✅ `toy2d_metrics.py` exits non-zero on failure |

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
permanently — extra steps cannot fix it. §4's "report per sampler (DDPM η=1 vs
DDIM η=0)" is therefore the next experiment and now has a specific hypothesis:
**η = 1 should even out the per-mode mass while leaving mode recall at 8/8.**
⛔ `Ddpm.ddimStep` cannot do it — η=1 needs the σ_t noise term added.

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
* Spiral, two-moons and checkerboard are generated but unused; only 8-gaussians
  is wired to metrics.

## 6. PHASES

| # | what | est. | gate |
|---|---|---|---|
| 0 | Data samplers for the four targets (host-side, `gaussN` + a few lines each) | S | a scatter of the *true* density looks right |
| 1 | `NetSpec` for the MLP denoiser + wire `prependSinCosT`'s scalar peer | S | shapes tie to `toSpecs`; render compiles |
| 2 | Train loop; reuse `stepInputs` unchanged | M | loss drops; **seconds per run** |
| 3 | Sample with the existing `ddimStep`; PPM/PNG scatter out | S | 8-gaussians visibly hits 8 modes |
| 4 | ⭐ The three metrics of §4 + the sampler × steps table | M | 8/8 mode recall, energy distance reported |
| 5 | The reverse-process strip (§5) | S | the figure |
| 6 | ⭐⭐ CI gate: mode recall = 8/8 and energy distance < threshold | S | runs in the corpus job, seconds |

▶ **Phase 6 is the payoff and should not be dropped.** It makes this the first
generative model in the repo with a regression test. Every image demo is
un-gateable by construction; this one is a handful of seconds and an integer.

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
- ⚠ **Class-conditional MNIST is a separate proposal** and a stronger one for
  the image side: condition on the digit, then score samples with the Chapter-3
  `cnnVerified` classifier (98.75 %) for a real number. That fixes the image
  demo's metric problem the same way §4 fixes it here. The two are
  complementary and neither blocks the other.

---

## 8. THE OPEN QUESTION

How many diffusion steps does a 2-D manifold actually need? The image demos use
50 DDIM steps because that is the convention, not because anything measured it.
With §4's metrics, the step-count sweep is a table rather than an opinion — and
if the answer on a spiral is "12," that is worth knowing before defending 50 on
an image where nobody can tell.
