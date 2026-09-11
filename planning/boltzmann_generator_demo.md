# boltzmann_generator_demo.md — flow matching as a Boltzmann generator

Goal: a second half for the diffusion demo whose ground truth is a formula.
Train the 2-D toy demo's MLP as a flow-matching model from `N(0, I)` to
`exp(-U/kT)` on the Müller-Brown surface, score it by quadrature instead of
against a second point cloud, and make the one claim MCMC cannot: samples
drawn once at kT = 20 give the free energies at kT = 12 and 8, where a
Langevin chain of 2×10^5 steps has not crossed the barrier.

Mock of the finished section (figure, both tables, captions), computed from
the physics alone on 2026-09-11: the "Boltzmann Generator & Ising NQS"
artifact. The numbers in it are what a perfect model would print; the trained
model's gap to them is this demo's headline.

Prerequisite reading: `planning/archive/diffusion_2d_demo.md` (§2 targets,
§4 metrics, §5.8 samplers, §7 scope — whose "not flow matching" line this doc
retires, on the terms it set: "adding rectified flow beside it later is cheap
*because* the metrics of §4 already exist").

## 0. The one-paragraph version

Same network, same rank-2 MSE block, same sincos time channel as
`demos/archive/MainDiffusion2d.lean`. The trainer changes one thing, the
interpolant: `x_t = (1-t)·x0 + t·ε` with target `ε - x0` instead of the
cosine-schedule `x_t` with target `ε`. The sampler changes one thing: Euler on
`dx/dt = v(x, t)` from t = 1 to 0 instead of DDIM. The target changes from a
point cloud to a density with a closed-form `U`, so populations, mean energy,
free-energy differences and the exact likelihood are all numbers by
quadrature on a grid. Zero new codegen. Every new line is host Lean, C
helpers of the `Ddpm.*` kind, and Python scoring.

## 1. Why this and not another image demo

- The DDPM section's instrument is a classifier; its floor is "real MNIST
  scored as if generated". Here the floor is exact and the ceiling is exact.
- Temperature is a physical knob the trained model can be moved along without
  retraining; nothing in the image demos has an analogue.
- It settles the sampler question §5.8 left open in the cleanest form: with
  the VP path (DDPM, already in the exe) and the linear path (this doc) on the
  same target, same net, same loss and same Euler integrator, the only thing
  that differs is the interpolant.
- Noé et al. 2019 (Science, "Boltzmann generators") is the paper; ours is the
  same claim on 18k parameters, with quadrature standing in for their
  importance-weighted free energies.

## 2. The target

Müller-Brown (Müller & Brown 1979), the standard three-well test surface for
reaction paths: `U(x,y) = Σ_i A_i exp(a_i dx² + b_i dx dy + c_i dy²)` with the
textbook constants (A = -200,-100,-170,15; a = -1,-1,-6.5,0.7; b = 0,0,11,0.6;
c = -10,-10,-6.5,0.7; x0 = 1,0,-0.5,-1; y0 = 0,0.5,1.5,1). Minima A
(-0.558, 1.442; U = -146.7), B (0.623, 0.028; -108.2), C (-0.050, 0.467;
-80.8); saddles at -40.7 and -72.2. The barrier out of B toward C is 36 units.

- Coordinates are standardised for the network: `z = (x - c)/s` with
  c = (-0.2, 0.75), s = 0.8, so N(0, I) covers the box [-1.7, 1.3] × [-0.7, 2.2].
  Same reason `preprocess_toy2d.py` scales its targets to unit radius.
- Training temperature kT = 20. Populations there are 0.806 / 0.129 / 0.065
  (A / B / C), so all three wells are visible in a 2048-point cloud.
- Transfer temperatures 12 and 8. At 8 the exact populations are
  0.991 / 0.009 / 0.001 and a chain started in B needs e^4.5 attempts per
  crossing.

Basins are assigned by gradient descent to a minimum, never by nearest
minimum: the wells are elongated and Voronoi cells put part of A's basin in C.

## 3. Data — the data is the physics

`preprocess_boltzmann.py [n] [outdir=data/boltzmann]`, numpy only:

- `mb_kT20.bin` — overdamped Langevin, `x ← x - ∇U dt + √(2 kT dt) ξ`,
  dt = 1e-4 (curvature in the deepest well is ~2200, so dt < 9e-4 for
  stability), 8 chains from uniform starts in the box, 10^5 steps each,
  10^4 burn-in, thinned by 20, in standardised coordinates. That is the
  Euler-Maruyama integrator §5.8 retired as a sampler, doing its actual job.
- `mb_kT20_ref.bin` — an independent draw from the quadrature density
  (multinomial over grid cells plus in-cell jitter), the energy distance's
  reference, matching `<name>_ref.bin`.
- `mb_grid.npz` — the 360×360 grid, `U`, the basin label per cell, and the
  exact populations, mean energy and ΔF at kT = 20, 12, 8.
- `manifest.json` — kT, c, s, minima, the constants; the scorer reads it
  rather than retyping.
- ⚠ The Langevin set is deliberately not equilibrated. At kT = 20, 8×10^5
  steps still over-weights A by four points (0.845 vs 0.806). The model
  trained on it will inherit that; the table shows it, and it is the reason
  the quadrature row exists.

## 4. The model and the trainer — zero new codegen

`demos/MainDiffusion2d.lean` comes back out of `demos/archive/` under its old
name and lake target `diffusion-2d`, with two additions:

1. A fifth target, `muller_brown`, whose data file comes from §3 and whose
   display name flows into `spec.name` and `buildPrefix` exactly as the other
   four do, so its MLIR, checkpoint and samples are keyed apart by name.
2. A `flow` flag. Training with it calls a new C helper
   `Ddpm.flowStepInputs` (per row: t ~ U(t_min, 1), `x_t = (1-t)x0 + t ε`,
   target `v = ε - x0`, and the integer index `round(t·Tmax)` for
   `prependSinCosT`, which stays untouched — the network's time channel is
   already a quantised index). Without the flag the exe is the DDPM
   baseline on the same target, which is the comparison row.

The train step is `generateTrainStep … (useDdpm := true) (ddpmOutShape :=
[B, 2, 1, 1])` and `LowererSession.trainStepAdamF32Ddpm`, unchanged: the MSE
block does not know whether the target is ε or v.

Sampler additions, beside `ddim / euler / heun / sde` in `Ddpm.samplerNfe`:
`fm-euler` (1 eval/step) and `fm-heun` (2), integrating `x ← x + v·Δt` from
t = 1 to t_min on a uniform grid. `strip` works for them once the frame
schedule is uniform in t rather than log σ (linear paths have no σ).

⭐ Compile the eval graph at batch `nGen`, not 1. The archived sampler runs
one point at a time because 18k params made it sub-second; the divergence
integration of §6 multiplies the evaluations by five and the field dump by a
grid, so `generateEval spec nGen` and one forward per solver step is the
change that keeps everything under a minute.

## 5. The instrument — `scripts/boltzmann_metrics.py`

Reads a samples `.bin`, the manifest and `mb_grid.npz`. Prints one row:

| column | how |
|---|---|
| p_A / p_B / p_C | basin by descent on the samples, vs quadrature |
| ⟨U⟩ | mean of the closed-form U over the samples |
| ΔF_AB = -kT ln(p_A/p_B) | from the populations |
| energy distance × floor | §4 of the 2-D plan, vs `mb_kT20_ref.bin`; the floor is exact-vs-exact (0.0006 in the mock) |
| KL to exact (phase 3) | needs the likelihood of §6 |

Bracket rows the table always carries: quadrature (top), Langevin training
set, the model at NFE 50 / 10 / 2, and the N(0, I) prior with no flow
(bottom, 737× the floor in the mock). At n = 1500 a population carries ±0.01
of sampling error; say so under the table rather than reading the third
decimal.

Regression gate (the 2-D plan's phase-6 pattern): energy distance ≤ 10× floor
at NFE 50 and every basin within 0.03 of quadrature. Set from the mock's
exact-field rows (8.1×), not tightened, for the reason the toy2d script gives:
nobody has measured retrain variance yet.

## 6. The physics claims, in order of cost

**6.1 Temperature transfer (host Python, no model change).** Reweight the
kT = 20 samples by `exp(-U (1/kT' - 1/kT))`, report populations, ΔF and the
effective sample size beside a fresh Langevin chain of 2×10^5 steps started in
B at kT'. Mock: ESS 1092 / 1500 at kT' = 12, 779 at 8; reweighted ΔF within
one unit of quadrature at both; Langevin off by 30 and 24 units. ⚠ This
version assumes the model's density is exactly `p_20`; the model-error
correction `p_20(x)/p_θ(x)` needs 6.3 and the difference between the two
reweightings is itself a row.

**6.2 Score the field, not the samples.** For any density the marginal
velocity is `v(x,t) = (x - E[x0 | x_t])/t` with the posterior
`∝ p(x0) N(x_t; (1-t)x0, t²I)`, a quadrature over the grid. A Lean `field`
mode dumps `v_θ` on a (grid × t) lattice through the batched eval graph; the
script reports the L² field error per t, plus two scalars: kinetic energy
`∫ E|v|² dt` and Liu's straightness `∫ E|v(x_t,t) - (x1 - x0)|² dt`. Both
should fall under 6.4.

**6.3 Exact likelihood by the continuity equation.** Along the ODE,
`d/dt log ρ = -∇·v`; integrate the divergence beside the state. In 2-D the
divergence is two central differences, four extra forwards per step through
the batched graph. Gives NLL on the grid against the known density, the KL
column of §5, and the corrected weights of 6.1. This is also the proofs'
hook: the divergence is the trace of the input Jacobian, and the
input-gradient ties already certify that object for `.dense` chains.

**6.4 Coupling and reflow (host only).** Two more training modes on the same
exe: `ot` pairs each batch's noise to data by a Hungarian assignment on the
host (Tong et al. 2023, minibatch OT), and `reflow` retrains on (noise,
sample) pairs drawn from the trained model's own ODE (Liu et al. 2022). Both
should straighten paths; the table is independent / OT / reflow × NFE
1 / 2 / 5 / 10 / 50 with the energy distance and the two scalars of 6.2.

**6.5 The interpolant row.** Train the DDPM path on `muller_brown` (the exe
without `flow`) and put its `euler` / `ddim` rows beside `fm-euler` at matched
NFE. Same net, loss, target and integrator; only the path differs.

## 7. Figure and section

`scripts/boltzmann_figure.py` (system python3 has matplotlib 3.10; the pinned
venv does not, same as the mock): three panels as in the artifact — the
surface with the Langevin training set, the flow's paths from noise into the
wells with hollow markers at t = 1, and the kT = 8 panel with the chain
started in B beside the model's samples and the three population rows in a
box. Series colours: blue for the model, orange for Langevin.

Section: a "Flow matching — demo" subsection after the DDPM one in
`blueprint/src/content.tex`, in the DDPM section's shape: three things that
change (the ground truth is a formula; a sample set is a free-energy
estimate; temperature is a knob), Figure, Table 1 with its lead-in and
caveat, Table 2 with its lead-in and caveat. Bestiary: `Bestiary/BoltzmannGenerator.lean` (Noé 2019: the original is a RealNVP
coupling flow, one new primitive if shown; the flow-matching version is zero)
with flow matching (Lipman 2022), rectified flow (Liu 2022) and stochastic
interpolants (Albergo & Vanden-Eijnden 2022) cited in prose. Add a
`data/boltzmann` row to the demo-datasets table with "generated by
`preprocess_boltzmann.py`, no download".

## 8. Phases

```
Phase 0 (½ session, CPU):   §3 preprocess + §5 scorer, validated on the exact draw
                             (must reproduce the mock's quadrature row)
Phase 1 (1 session):         §4 un-archive + flowStepInputs + fm samplers + batched eval;
                             Gate A: fm-euler NFE 50 within the §5 regression gate
Phase 2 (½ session):         6.1 transfer table + 6.5 interpolant row;
                             Gate B: reweighted ΔF within 2 units of quadrature at kT' = 12
Phase 3 (1 session):         6.2 field dump + 6.3 divergence/likelihood; KL column lands
Phase 4 (½ session each):    6.4 OT coupling, then reflow; the NFE 1 / 2 rows
Phase 5 (½ session):         §7 figure + section + bestiary entry
```

Every run is seconds to minutes on one card at 18k params; no run needs
asking about. Phase 0 needs no GPU at all.

## 9. Gates that fail loudly

- The scorer run on `mb_kT20_ref.bin` must print the quadrature populations
  to ±0.01 and an energy distance at the floor. If it does not, the basin
  assignment or the standardisation is wrong, and nothing downstream means
  anything.
- The kT = 8 Langevin control must over-weight B. If a fresh chain reaches
  the quadrature populations in 2×10^5 steps, dt or the temperature is not
  what §2 says.
- NFE 2 must be visibly wrong (mock: well B empties, ⟨U⟩ off by 20) while
  its populations look plausible — the DDPM table's Euler-Maruyama lesson,
  and the reason ⟨U⟩ and ΔF are columns.

## 10. Out of scope

Molecules and equivariance (FermiNet, NequIP: new primitive families);
Hamiltonian networks and PINNs (second-order training the emitter lacks);
lattice field theory (same method, bigger target; a one-pager if this lands);
any CIFAR-scale pixel version (`ddpm_demo_v3.md`'s backlog, separate).

## 11. Old-code notes

The archived exe's `logsnr` / `logabar` grids are VP-schedule concepts and do
not apply to the linear path; the flow samplers use a uniform grid and the
flag is refused with them, the way `strip` is refused for non-DDIM samplers.
`Ddpm.sampleNoise` was fixed on 2026-08-28 (§5.7 of the 2-D plan); the flow
helper must use the same Box-Muller path, not a new one.
