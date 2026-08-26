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
