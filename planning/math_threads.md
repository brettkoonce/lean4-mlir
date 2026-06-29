# Planning — Math threads worth exposing (research/book directions)

Generative math directions for the book, beyond the (technically-valid but cookie-cutter)
ablation grind. The thesis: **the formalization is already deep math wearing work clothes** —
`pdiv = fderiv` + `HasVJP` is the cotangent pullback; `FloatClose` is Higham rounding theory;
`sgd_descends` is optimization theory. The highest-leverage move is half "surface the math
already holding the proofs up" and half "add a few generative threads that extend a framework."
Each thread below: the math core, where it already lives in the repo, the verification angle,
and a priority. Companion to `planning/muon.md` (Muon is Thread 1's first landed point).

## 0. TL;DR

- **Thread 1 (headliner): optimizer = steepest descent under a metric.** Muon is one point on a
  ladder: SGD (identity) → Adam (diagonal) → **Muon (spectral)** → natural gradient (Fisher) →
  Shampoo/K-FAC (Kronecker). The jewel: **single-step Shampoo = Muon = the polar factor `UVᵀ`,
  exactly.** One framework, the whole optimizer zoo, and the chapter's "optimizers + convergence"
  theme finally has a spine. They share a verification kernel (NS + the diagonalization lemma).
- **Thread 2: the orthogonality through-line.** Spectral/orthogonal control at init (dynamical
  isometry), in training (Muon), and in normalization (whitening) — same "kill the anisotropy"
  idea on the Jacobian / the update / the activations. Connective tissue for the back half.
- **Thread 3: surface the cotangent geometry (nearly free).** Backprop = adjoint pullback;
  ResNet = discretized ODE & its backward = the adjoint ODE; attention = kernel smoothing,
  softmax = ∇log-sum-exp. Reframes existing proofs as the geometry they already are.
- **Thread 4 (new dimension): Lipschitz / spectral-norm certification.** The repo has ZERO
  robustness content but has conv + attention; their spectral norms compose into a *provable*
  ε-robustness certificate — new math, on home turf, and it plugs the gap an external reviewer
  flags first.

Recommended sequencing: build Thread 1 as the next real chapter; run Thread 2 as the spine;
narrate Thread 3 throughout (the antidote to "formalization feels mechanical"); Thread 4 when
you want a result a non-formalization audience values instantly.

## 0.5 Demo-first protocol (the Muon-proven workflow)

**Build the runnable demo first; work backwards to the formalization later.** Muon validated
this: perf-path codegen → an A/B curve on real hardware → *then* the `den=` tie. The empirical
signal tells you which math survives contact and is worth the proof effort; the formalization is
the rigor pass on what already works, not the driver. So for every thread:

1. **Demo** — emit it on the UNVERIFIED perf path (`MlirCodegen.lean`), wire a trainer, run an A/B
   on the 2×gfx1100 box. Reuse the Muon harness (`apps/baselines/Main*Train.lean`,
   `run_*_ab.sh`, the `.venv/bin/iree-compile` symlink — see `muon-scaffold`).
2. **Signal** — the curve / number. Keep or kill the idea here, cheaply.
3. **Formalize later** — only the survivors get the `den=` render-tie / convergence lemma. The
   shared verification kernel (NS + the diagonalization lemma, `muon.md §8.4`) means one proof
   often covers several threads.

Each thread below tags its **First demo** (cheapest runnable probe) + effort.

---

## Thread 1 — Optimizer = steepest descent under a metric (the headliner)

### The framework (from `muon.md §8.1`, now the organizing principle)

Every first-order optimizer is `ΔW = argmin_Δ ⟨G,Δ⟩ + (1/2η)‖Δ‖²`. The **metric** on `Δ` is the
optimizer's whole identity:

| metric on Δ | preconditioner | optimizer |
|---|---|---|
| identity `I` | — | SGD |
| diagonal `diag(√v̂)` (empirical Fisher diag) | per-coordinate | Adagrad / Adam / RMSprop |
| full-matrix `(Σ ggᵀ)^{1/2}` | dense (intractable) | full-matrix Adagrad |
| Kronecker `(Σ GGᵀ)^{1/4} ⊗ (Σ GᵀG)^{1/4}` | two small roots | **Shampoo** |
| Fisher `F = E[∇logp ∇logpᵀ]` | natural metric (KL geometry) | **natural gradient** (Amari) |
| Kronecker Fisher `A ⊗ S` | `S⁻¹ G A⁻¹` | **K-FAC** |
| spectral `‖Δ‖₂` | polar factor `UVᵀ` | **Muon** |

### The jewel: Muon = single-step Shampoo = the polar factor

Shampoo (matrix param `W∈ℝ^{m×n}`): `L = Σ_t G_t G_tᵀ`, `R = Σ_t G_tᵀ G_t`, update
`W ← W − η L^{−1/4} G R^{−1/4}`. Take a SINGLE gradient (no accumulation), `L=GGᵀ`, `R=GᵀG`,
and use `G = UΣVᵀ`:

```
(GGᵀ)^{−1/4} = U Σ^{−1/2} Uᵀ ,   (GᵀG)^{−1/4} = V Σ^{−1/2} Vᵀ
L^{−1/4} G R^{−1/4} = U Σ^{−1/2} Uᵀ · UΣVᵀ · V Σ^{−1/2} Vᵀ = U Σ^{−1/2}·Σ·Σ^{−1/2} Vᵀ = U Σ⁰ Vᵀ = U Vᵀ.
```

So **un-accumulated Shampoo IS Muon IS the polar factor**, exactly. Accumulated Shampoo is a
*memory* knob interpolating between full-matrix Adagrad (long history) and Muon (one step). That's
the through-line from your 2017 K-FAC/Shampoo to the 2024 frontier optimizer: **same operator,
different accumulation + different root power.** (Bernstein–Newhouse "Old Optimizer, New Norm"
formalizes the norm side; the SVD identity above is the cleanest one-liner.)

### Natural gradient → K-FAC (the curvature side)

Natural gradient: `θ ← θ − η F⁻¹∇L`, `F` = Fisher = the Hessian of KL ⇒ steepest descent in the
distribution's own geometry (invariant to reparametrization). K-FAC makes `F⁻¹` tractable by the
Kronecker factorization `F ≈ A ⊗ S` (`A` = E[aaᵀ] input-activation cov, `S` = E[ggᵀ]
pre-activation-gradient cov), so the per-layer natural-gradient step is `S⁻¹ (∇_W L) A⁻¹` — two
small solves. Shampoo is the same Kronecker move on the *gradient second-moment* (Adagrad)
instead of the Fisher.

### Why this is a great FORMALIZATION target (not just book content)

Every optimizer here is **matmul + a matrix inverse-root** (`(·)^{−1/4}`, `(·)^{−1/2}`). Those
roots are computed in practice by **Newton–Schulz / coupled-Newton iterations** — the SAME
machinery as Muon. So:
- **Render-faithful (`den=`):** the whole family is `dot_general` + NS ⇒ reachable, same tech as
  the Muon tie (`planning/muon.md §8.6`).
- **NS-convergence deep cut generalizes:** by the diagonalization lemma (`muon.md §8.4`) the
  matrix-root iteration reduces to a *scalar* per-eigenvalue iteration. ONE Mathlib lemma about a
  scalar Newton iteration for `x^{−p}` covers Muon's `(·)^{−1/2}` AND Shampoo's `(·)^{−1/4}` AND
  K-FAC's inverse. Formalizing one buys the family.
- **Descent:** open for all (spectral/Fisher-norm trust-region argument), same honest frontier.

Where it lives now: `OptimizerKind` (`Types.lean`), the `emit*Update` family + `emitMuonUpdate`
(`MlirCodegen.lean`), `SgdDescent*.lean` (the descent proofs none-but-SGD fit yet).

**Landed + concretized:** Muon geometry L1–L6 (`MuonGeometry.lean`) + Newton–Schulz convergence P1–P4
(`MuonNewtonSchulz.lean`, cubic & principled-quintic matmul iterate → `UVᵀ`). The capstone that finishes
this ladder is now a concrete plan: **`planning/natural_gradient.md`** — steepest descent under a metric
`M` generalizes `steepest_l2_*`; Adam = diagonal `M`, **natural gradient = Fisher `M = JᵀJ`** (built from
the VJP machinery), and Adam/Shampoo/Muon become structured approximations to `F⁻¹g`. The unifying
NG1 lemma makes every existing rung a corollary.

**First demo:** `OptimizerKind.shampoo` + `emitShampooUpdate` (matmul + the two NS inverse-roots),
`vit-tiny-shampoo-train`, A/B vs Muon + AdamW — the **Muon harness is the template** (clone
`MainVitMuonTrain.lean` + `run_vit_muon_ab.sh`). The one new wrinkle: Shampoo needs `L`(m×m)/`R`(n×n)
matrix state, so it CAN'T reuse Muon's m/v-slot trick — the optimizer-state shape changes
(`emitTrainStepSig` + host allocation). That's the only real lift over Muon.
**Effort:** Medium (state-shape change is the cost). The book chapter is the bigger, cheaper win;
the codegen demo is a strong second since it directly extends Muon and gives a 3-way A/B.

---

## Thread 2 — The orthogonality through-line (connective tissue)

Spectral/orthogonal control shows up at every stage, un-narrated:
- **Init — dynamical isometry.** Orthogonal init makes the input→output Jacobian's singular
  spectrum ≈ 1, so signal neither vanishes nor explodes through depth (Saxe et al. 2013;
  Pennington–Schoenholz–Ganguli 2017). You use He; orthogonal is the spectral cousin.
- **Training — Muon.** Flatten the *update's* spectrum (`UVᵀ`).
- **Normalization — whitening.** LN/BN/ZCA flatten the *activations'* spectrum.

Same instinct (kill anisotropy) on the **Jacobian**, the **update**, the **activations**. Run it
as a recurring motif through init / optimizer / normalization chapters. Seeded already by your
"LN vs BN = different axes" intuition. Where it lives: He init in `VerifiedTrain.lean`/codegen,
BN/LN VJPs (`BatchNorm`/`LayerNorm` proofs), Muon.
**First demo:** add orthogonal init (host-side, a QR of a Gaussian — the init runs in Lean before
codegen, so NO new MLIR), A/B He-init vs orthogonal-init on ViT/ResNet. Cheapest demo of the lot.
**Effort:** Low (init function + an A/B). Pure narrative otherwise.

---

## Thread 3 — Surface the cotangent geometry (the "it was deep math all along" thread)

The antidote to "formalization feels mechanical." Three self-contained gems already in code:

1. **Backprop = the cotangent pullback.** `pdiv = fderiv`, `HasVJP.correct : backward = Σ pdiv·dy`
   is literally the loss covector pulled back through the cotangent bundle (forward-mode = push
   tangents; reverse-mode = pull cotangents). Categorically: backprop-as-functor / lenses
   (Fong–Spivak–Tuyéras 2019). Your `vjp_comp` is functoriality.
2. **ResNet = discretized ODE; its backward = the adjoint ODE.** `x + f(x)` is forward Euler;
   depth = time; your VJP through the residual chain IS the adjoint method (Chen et al., Neural
   ODEs 2018). Explains *why* residuals train deep (block Jacobian ≈ `I + εJ`, stays near
   identity) and reframes `resblock_has_vjp` as continuous-time dynamics.
3. **Attention = kernel smoothing; softmax = ∇log-sum-exp.** Softmax is the gradient of the convex
   LSE (the conjugate of negative entropy on the simplex); attention is Nadaraya–Watson kernel
   regression. Your `softmax_perturb` Lipschitz lemma IS the kernel-stability bound.

Where it lives: `Tensor.lean` (`pdiv`/`HasVJP`), `Residual`/`ResNet34` proofs, `Attention.lean`.
**First demo:** none — this thread is *narration of existing code*, not a new mechanism. (If you
want a toy: a Neural-ODE block as a `.residual` with shared weights across N steps, A/B vs a plain
ResNet block — shows depth=time empirically.)
**Effort:** narrative + sidebars; zero new proofs (it re-describes existing ones). Do it inline as
each chapter goes by.

---

## Thread 4 — Lipschitz / spectral-norm certification (the new dimension)

The repo has **no robustness content** (an external reviewer notices immediately — it's a
"verified DL" book with no certificate over an input region). But you have the pieces:
- **Conv spectral norm exactly** via the FFT of the kernel (the conv operator is block-circulant;
  Sedghi–Gupta–Long 2019 give the exact singular values).
- **Dense/attention Lipschitz** (attention's is subtle — Kim–Papamakarios–Mnih 2021).
- **Composition:** `Lip(f∘g) ≤ Lip(f)·Lip(g)` ⇒ a global Lipschitz bound ⇒ a **provable
  ε-robustness certificate** (`‖f(x+δ)−f(x)‖ ≤ L‖δ‖`), the real version of what the audit's framing
  expected and didn't find.

This is the one thread that is BOTH genuinely new math AND produces something a non-formalization
audience values on sight ("certified robust to ε-perturbations"), AND it's your conv home turf
(Toeplitz/circulant spectral analysis). Verification angle: a real `Lipschitz f L` predicate +
per-layer spectral-norm bounds composing — provable, non-vacuous if the bounds are tight (the
honest catch: naive products are loose; tightness is the research).
**First demo:** a PGD adversarial attack on a trained net vs the **certified ε** (compute the
spectral-norm product as a number, attack, show the attack can't beat the certificate). A real
runnable demo a non-formalization audience reads instantly. Host-side mostly (attack loop +
spectral-norm via FFT/power-iteration), light codegen.
**Effort:** the largest to FORMALIZE (new predicate + per-op spectral-norm lemmas), but the demo
is medium and the biggest "this isn't cookie-cutter" payoff. Pairs with Thread 2 (all spectral norms).

**Landed + concretized:** the cert is a theorem (`LipschitzCert.lean`: `clm_lipschitzL2` +
`LipschitzL2.comp` ⟹ `∏‖Wᵢ‖₂`; `lipschitz_margin_certified_radius`; randomized-smoothing
`smoothing_certified_radius`). The open *computational* half — actually *producing* the `‖Wᵢ‖₂` the
cert assumes — is now a concrete plan: **`planning/power_iteration_lipschitz.md`**. Power iteration is
a Newton–Schulz cousin that reuses `conj_diag_pow` wholesale (the spectrum reduces to a scalar ratio
`(λ₂/λ₁)ᵏ`), so it's the bridge from Thread 1's spectral engine to this cert. Honest catch carried over:
it converges from *below*, so the sound *upper* bound for a finite-step cert is the separate harder rung.

---

## Sequencing (demo-first)

Demos to get empirical signal, formalization backwards from the survivors (per §0.5):

1. **Shampoo demo (Thread 1)** — clone the Muon harness, get a 3-way A/B (AdamW vs Muon vs Shampoo).
   The most direct extension of what's already running; only new cost is the L/R matrix state.
2. **Orthogonal-init demo (Thread 2)** — cheapest of all (host-side init + an A/B), and seeds the
   orthogonality spine.
3. **Robustness demo (Thread 4)** — PGD-vs-certificate; the standout "new math" result, plugs the
   robustness gap, and is the one a general audience values on sight.
4. **Thread 3** — no demo; narrate the cotangent geometry inline as each chapter goes by.

Formalization (the "nonsense" 😄) comes after a demo earns it — and the shared NS+diagonalization
kernel means Thread-1's proof largely covers Shampoo/K-FAC too. The book chapters can be written
from the demos + the math (Thread 1 §§ above) well before the proofs land.

## References

- Amari, *Natural Gradient Works Efficiently in Learning* (1998).
- Martens & Grosse, *Optimizing Neural Networks with Kronecker-factored Approximate Curvature*
  (K-FAC, 2015).
- Gupta, Koren, Singer, *Shampoo: Preconditioned Stochastic Tensor Optimization* (2018).
- Bernstein & Newhouse, *Old Optimizer, New Norm* / modular norms (2024) — the norm framing.
- Jordan, *Muon* (2024). Higham, *Functions of Matrices* (the matrix-root NS family).
- Saxe, McClelland, Ganguli (2013) + Pennington, Schoenholz, Ganguli (2017) — orthogonal init /
  dynamical isometry.
- Chen, Rubanova, Bettencourt, Duvenaud, *Neural ODEs* (2018) — adjoint = backprop.
- Fong, Spivak, Tuyéras, *Backprop as Functor* (2019).
- Sedghi, Gupta, Long, *The Singular Values of Convolutional Layers* (2019); Kim, Papamakarios,
  Mnih, *The Lipschitz Constant of Self-Attention* (2021).
- In-repo: `planning/muon.md` (Thread 1's landed first point), `Tensor.lean` (`pdiv`/`HasVJP`),
  `OptimizerKind`/`MlirCodegen.lean` (the optimizer emit family), `SgdDescent*.lean`,
  `Attention.lean`, `BatchNorm`/`LayerNorm`/`Residual` proofs.
