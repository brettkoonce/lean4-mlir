# transformer_wavefunction_demo.md — structure first, the transformer models the rest

Goal: the science demo. Ground states of spin chains as networks trained
through the stack, with one design rule: the ansatz is a reference state
physics already knows times a learned residual, and the residual model is a
transformer — a ViT encoder first, then a GPT decoder whose sampling loop is
TinyGPT's. Every rung is exactly scorable: enumeration at N = 12, the
Jordan-Wigner solution of the Ising chain at any N, and the Majumdar-Ghosh
point of the J1-J2 chain.

Mock of the finished section's shape (figure and table, real numbers at
N = 12): the "Boltzmann Generator & Ising NQS" artifact, 2026-09-11. Its RBM
rows are history and do not carry over; the ladder here is mean field, MLP
residual, ViT residual, GPT, exact.

## 0. The one-paragraph version

Write ψ_θ(σ) = ψ_ref(σ) · exp f_θ(σ). The reference is a closed-form state
(the mean-field product state for the Ising chain, the Marshall sign rule
for the Heisenberg chain), so f_θ = 0 *is* the floor row of the table and the
network only has to model what the reference gets wrong. That is Chapter 5's
identity-plus-residual move applied to a wavefunction, and it is what should
keep the optimiser at Adam: the mock's wider ansatz started from the uniform
state and stalled in the polarised state at small field; started from the
mean-field state there is nothing to stall into. The energy is a weighted
mean over configurations, its gradient is one host weight per configuration
(two with a phase), and the existing rank-2 MSE block carries it: zero new
codegen through rung 3. The transformer is the Chapter 9 kit on spin tokens,
and its VJP is the one already proved.

## 1. The design rule, precisely

- Real ansatz (rungs 1–3): log ψ_θ = log ψ_ref + f_θ. The host adds
  log ψ_ref(σ) to the network's output before any ratio is formed; the
  network never sees the reference.
- With a phase (rung 4): log ψ_θ = log ψ_ref + f_θ + i·φ_θ, a two-slot head.
  The reference carries the sign rule; φ_θ models the deviation from it,
  which is zero at J2 = 0 and grows with frustration.
- Autoregressive ansatz (rung 3): normalisation must survive, so the
  reference enters as a fixed bias on the logits: p(patch_k | <k) =
  softmax(logits_θ + log p_ref(patch_k)). At f_θ = 0 the conditionals are the
  reference's; the host applies the softmax to 2^p entries per token.
- References. Ising chain: the product state |φ*⟩^N with the optimal angle,
  E_MF/N = min_φ [−J cos²φ − h sin φ], whose log-amplitude is linear in σ
  (a fixed visible bias, nothing to learn). J1-J2 chain: the Marshall sign
  (−1)^(number of up spins on one sublattice), exact at J2 = 0, times a
  two-body Jastrow if wanted; at J2 = J1/2 the Majumdar-Ghosh dimer product
  is a second, exact reference.

**Table 2 exists to test the rule**: the same ViT, Adam, same steps, from the
uniform start and from the reference, at h = 0.2 and 0.4 where the uniform
start stalled. "Structure buys X orders of magnitude at small field" is the
sentence, and the table decides X.

## 2. Models and their exact instruments

**Transverse-field Ising chain**, periodic, H = −J Σ σᶻᵢσᶻᵢ₊₁ − h Σ σˣᵢ.
Amplitudes are positive in the z basis, so rungs 1–3 are real.

- N = 12: everything by enumeration of 4096 configurations; E0 from
  `eigvalsh` of the 4096 × 4096 Hamiltonian, as in the mock.
- N = 64: E0 exact by Jordan-Wigner, the free-fermion sum with the
  even-parity sector's antiperiodic momenta; ⟨σˣ⟩ likewise; the correlation
  ⟨σᶻᵢσᶻᵢ₊ᵣ⟩ by the Toeplitz determinant of Lieb, Schultz & Mattis 1961 and
  Pfeuty 1970, a numpy determinant of size r. Gate: the N = 12 values of all
  three must match enumeration before any N = 64 number is quoted.
- Var(E_loc) at any N, from the samples alone: zero for an eigenstate, the
  instrument that needs no ceiling.

**J1-J2 Heisenberg chain** (rung 4), H = J1 Σ Sᵢ·Sᵢ₊₁ + J2 Σ Sᵢ·Sᵢ₊₂.
Frustrated for J2 > 0, so the ground state has a sign structure.

- N ≤ 20: Lanczos in the S_z = 0 sector (scipy, 184k states at N = 20).
- J2 = J1/2 at any even N: E0/N = −3/8 J1 exactly (Majumdar & Ghosh 1969).
- J2 = 0: the Marshall sign is the exact sign; the phase head must learn 0.

## 3. The rungs

**R0, the floor.** The mean-field product state, evaluated exactly. A row,
not a model.

**R1, MLP residual, N = 12.** `.dense 12 64 .relu, .dense 64 64 .relu,
.dense 64 1 .identity` on σ ∈ {±1}^12, `imageH := imageW := 1`, plus the
reference. Enumeration, so energy and gradient are exact and the first
training curve has no Monte Carlo noise in it.

**R2, ViT residual.** Tokens are patches of p spins, the patch state as an
integer id: `.tokenPositionEmbed (2^p) (N/p) d (idsInput := true)`,
`.transformerEncoder d heads mlpDim L (keepSequence := true)`, a mean over
tokens (`spatialUnflatten d (N/p) 1` then `.globalAvgPool`), `.dense d 1
.identity`. Real parameters throughout. N = 12 first with p = 2 (six tokens
of vocabulary 4, still enumerated), then N = 64 with p = 4 (sixteen tokens of
vocabulary 16) and Metropolis sampling: B parallel chains, single-spin flips,
acceptance from ψ² ratios read off the batched eval graph, a few sweeps
between samples. d = 32–64, two to four blocks, ~10^4 params.

**R3, GPT wavefunction.** Same tokens, `.transformerEncoder … (causalMask :=
true)`, `.lmHead d (2^p) (N/p)`. The conditionals are the network's output:
|ψ(σ)|² = Π_k p(patch_k | patch_<k), log ψ = ½ Σ log p for the positive
ansatz. Sampling is exact and independent, patch by patch through the
batched eval graph — the `tinygpt-shakespeare sample` loop with a
vocabulary of 16 and a sequence of 16, reused as is. No Metropolis anywhere.
This is the rung where N = 64 with an exact ceiling is the point.

**R4, the sign-structure rung.** J1-J2 chain, two-slot head (log|ψ|, φ) on
the ViT of R2, reference = Marshall sign. The GPT variant keeps its
conditionals for |ψ|² and reads φ from a second head on the final tokens
(the pattern of Hibat-Allah et al. 2020 and Sharir et al. 2020). Rows at
J2/J1 = 0, 0.25, 0.5 with and without the sign prior; ED ceiling at N = 16
or 20; the exact −3/8 at J2 = J1/2.

**R5, optional: stochastic reconfiguration.** Only if Adam with structure
leaves a visible gap at N = 64. SR needs per-configuration gradient vectors,
which the train step does not return; it is an emitter variant that leaves
the batch axis on the weight gradients ([B, …], 40 MB at B = 1024 and 10^4
params), then a host Woodbury solve of size B (the identity of Chen & Heyl
2024 and Rende et al. 2024). One session of codegen; stated so that the
decision is a cost, not a hope.

## 4. Mechanics through the stack

- Eval graph at batch M via `generateEval spec M`; every ψ evaluation goes
  through it in one call.
- E_loc for the Ising chain: E_loc(s) = −J Σ sᵢsᵢ₊₁ − h Σᵢ ψ(flipᵢ s)/ψ(s),
  so a batch of B samples needs ψ at B·N flipped configurations: one forward
  at batch B·N (65k at N = 64, B = 1024; small for a 10^4-param net). The
  Heisenberg chain replaces flips with the bond exchanges that H connects.
- Gradient: ∂E/∂θ = 2 Σ_s p_s (E_loc(s) − E) ∂_θ log ψ(s). With sampled s
  the weights are w_s = 2 (E_loc(s) − E)/B; with enumeration p_s is exact.
  Hand w to the train step through the rank-2 MSE block, `ddpmOutShape :=
  [B, 1, 1, 1]`: set the target to `out − B·w/2` so the block's gradient is
  exactly w. With a phase, `[B, 2, 1, 1]` and the two weights
  2 p Re(E_loc − E) and 2 p Im(E_loc − E). For the GPT the host chains the
  softmax Jacobian on 2^p entries per token before handing the logit
  cotangent over. If the extra forward ever matters, a ten-line "linear
  loss" branch whose gradient is its target replaces the trick.
- Samples: enumeration at N = 12; Metropolis for R2 at N = 64;
  autoregressive for R3. Every sampler writes `.lake/build/nqs_samples_*.bin`
  and the scorer does not care which produced them.
- Inputs: ±1 floats for the MLP; patch ids for the transformer, built on the
  host from the configuration.

## 5. Instrument and bracket — `scripts/nqs_metrics.py`

Rows: mean field (floor), MLP residual, ViT residual, GPT, exact (ceiling).
Columns: (E − E0)/|E0|, Var(E_loc), ⟨σˣ⟩, ⟨σᶻ₁σᶻ₁₊ᵣ⟩ at r = N/2. At N = 12
the exact column is enumeration; at N = 64 it is Jordan-Wigner and Pfeuty.
Regression gate: relative error ≤ 10⁻⁴ at h = J for the ViT residual at
N = 12 (the mock's converged rows sat at 10⁻⁵), variance ≤ 10⁻².

## 6. Figure, tables, section

**Figure.** (a) The ansatz as structure times residual: spins, the
reference's fixed bias, the tokens, the transformer, the head, and where
the host adds them. (b) Relative energy error against h for the ladder, at
N = 12 and N = 64 side by side. (c) The long-range test: ⟨σᶻ₁σᶻ₁₊ᵣ⟩ against r
at h = J, N = 64, exact against ViT against GPT — the panel attention is
supposed to win.

**Table 1**, the ladder at h = J. **Table 2**, the structure ablation of §1.
**Table 3** (R4), J1-J2 with and without the sign prior.

**Section.** "Beyond vision — science demo: neural quantum states" in the
demo shape: three things that change (there is no dataset; the loss is an
energy and the target is the model's own ratios; the instrument is a solved
model), the figure, Tables 1–2. **Bestiary**:
`Bestiary/NeuralQuantumState.lean` with `vitWavefunction` and
`gptWavefunction` as specs, and the RBM of Carleo & Troyer 2017, the CNN and
ResNet wavefunctions, Sharir 2020, Viteritti-Rende-Becca 2023, Zhang &
Di Ventra 2023 and Sprague & Czischek 2024 in prose.

**Proof item**, optional and cheap: the energy-gradient identity of §4 as a
Lean theorem over finite sums for real ψ — the loss's own VJP theorem, the
way the per-pixel CE and Dice blocks have theirs.

## 7. Phases

```
Phase 0 (½ session, CPU):    Jordan-Wigner + Pfeuty script, gated against enumeration at N = 12
Phase 1 (½ session):         R1 MLP residual by enumeration; Table 2 at N = 12 (structure ablation)
Phase 2 (1 session):         R2 ViT residual at N = 12, then N = 64 with Metropolis; Figure (b)
Phase 3 (1 session):         R3 GPT wavefunction, TinyGPT sampler reused; Figure (c), Table 1 at N = 64
Phase 4 (1 session):         R4 J1-J2 with the phase head; Table 3
Phase 5 (½ session):         section + bestiary + the proof item
Phase 6 (optional):          R5 stochastic reconfiguration
```

All runs are seconds to minutes on one card at 10^4 params; nothing needs
asking about.

## 8. Gates that fail loudly

- Phase 0's closed forms must reproduce enumeration at N = 12 to 10⁻¹⁰.
  Every N = 64 number rests on them.
- R1 at f_θ = 0 must print the mean-field row exactly. If it does not, the
  host is not adding the reference where §4 says.
- R3's samples must give ⟨σˣ⟩ within Monte Carlo error of the same
  wavefunction's enumerated value at N = 12. That is the check that exact
  sampling is exact, and it is the TinyGPT sampler's correctness test too.
- The variance must flag any stall before the energy column does; if a row
  has low variance and a wrong energy, it has converged to an excited state,
  which is a different failure and gets said.

## 9. Out of scope

Two-dimensional lattices (the 10×10 J1-J2 results need SR and symmetry
projection, a week each); fermions (FermiNet, determinants); time evolution
and open systems; pretraining on quantum-simulator snapshots (Lange et al.
2024), which is the natural extension once R3 exists, since it is the
language-model recipe on the GPT wavefunction.
