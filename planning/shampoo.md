# Planning — Shampoo demo (the next optimizer-as-metric point)

Demo-first scaffold for **Shampoo** (Gupta–Koren–Singer 2018), the second landed point on the
optimizer-as-metric ladder (`planning/math_threads.md` Thread 1; `planning/muon.md` was the first).
Shampoo is a Kronecker-factored full-matrix preconditioner — and its single-step limit is exactly
Muon, which is the connection worth demoing.

## 0. TL;DR

- **What.** For a 2D weight `W∈ℝ^{m×n}`, accumulate `L = Σ GGᵀ` (m×m) and `R = Σ GᵀG` (n×n), and step
  `W ← W − η · L^{−1/4} · G · R^{−1/4}`. A preconditioner with *memory* of past gradients.
- **The jewel.** Single-step (un-accumulated) Shampoo = Muon = the polar factor `UVᵀ`:
  `(GGᵀ)^{−1/4} G (GᵀG)^{−1/4} = U Σ^{−1/2}·Σ·Σ^{−1/2} Vᵀ = UVᵀ`. Accumulated Shampoo interpolates
  Adagrad (long memory) ↔ Muon (one step). Demo this and the ladder has its spine.
- **Feasibility (scoped 2026-06-28).** Lean-only — **NO C FFI change, NO `.so` rebuild** (the FFI
  iterates an arbitrary buffer list; `ffi/iree_ffi.c:298-321`). The one real lift: Shampoo's state
  is `L[m,m]`/`R[n,n]` — **not param-shaped**, so it breaks the host's "weights++m++v as three equal
  `nP` blocks" assumption (Muon dodged all state plumbing by reusing the m/v slots).
- **Effort.** Medium (a few focused hours). NS math + trainer are Muon templates; the cost is
  threading non-param-shaped state through the codegen signature AND the host packing/slicing.
- **Plan.** Numerical jewel demo first (cheap, validates math + the ε/warmup behavior), then the ViT
  A/B integration. Render-faithful `den=` tie is future work and shares Muon's NS-convergence kernel.

## 1. The math

Shampoo for a matrix parameter (the case we have — every Muon-routed 2D weight):

```
L_t = L_{t-1} + G_t G_tᵀ            # m×m, left preconditioner accumulator
R_t = R_{t-1} + G_tᵀ G_t            # n×n, right
W  ← W − η · L_t^{−1/4} · G_t · R_t^{−1/4}
```

The `−1/4` (not `−1/2`): `L` and `R` are each one Kronecker factor of the full
`(Σ vec(G)vec(G)ᵀ)^{1/2}` preconditioner; splitting the `1/2` across the two factors gives `1/4`
each. Optional EMA form `L_t = βL_{t-1} + (1-β)GGᵀ` (Adam-style memory) — what most modern impls use.

**Single-step = Muon (the jewel).** With no accumulation (`L=GGᵀ`, `R=GᵀG`) and `G=UΣVᵀ`:
`(GGᵀ)^{−1/4} = UΣ^{−1/2}Uᵀ`, `(GᵀG)^{−1/4} = VΣ^{−1/2}Vᵀ`, so the update is
`UΣ^{−1/2}Uᵀ·UΣVᵀ·VΣ^{−1/2}Vᵀ = UΣ^{0}Vᵀ = UVᵀ` — Muon's polar factor. Accumulation is the only
thing separating them; it's a *memory* knob.

**Computing `L^{−1/4}` (matmul-only, reuses Muon's NS infra).** `L` is symmetric PSD, `L=QΛQᵀ`. By
the diagonalization lemma (`muon.md §8.4`, symmetric version) any Newton–Schulz polynomial acts on
the eigenvalues: the matrix iteration for `L^{−1/4}` reduces to the *scalar* iteration `λ ↦ λ^{−1/4}`
per eigenvalue — pure `dot_general`. Practically: an inverse-square-root NS (coupled Newton) gives
`L^{−1/2}`, applied twice (or a direct inverse-4th-root coupled iteration). Unlike Muon's polar NS
(Frobenius-normalized), inverse-root NS **needs eigenvalue scaling** (divide `L` by `tr(L)` or a
`λ_max` estimate) to converge — a few more iterations.

## 2. Implementation scope (all Lean — no C)

| where | change | effort |
|---|---|---|
| `Types.lean` | `OptimizerKind.shampoo` + `TrainConfig.useShampoo` (mirror `useMuon`) | trivial |
| `MlirCodegen.lean` `emitShampooUpdate` | the accumulate + `L^{−1/4} G R^{−1/4}` update; NS inverse-roots reuse the Muon NS emit | medium |
| `MlirCodegen.lean` `emitTrainStepSig` (`:6807`) | **emit the `L`/`R` state args** per 2D weight (NEW: non-param-shaped buffers, `[m,m]`/`[n,n]`, beyond the param-shaped m/v) + return updated L/R | **the real lift** |
| `MlirCodegen.lean` dispatch closure (`:6412`) | route 2D weights → Shampoo (same `is2DMuon` test) | trivial |
| `Train.lean` run loop | build/init `L`,`R` (εI), append to `packed` (`:601`), extend `allShapes` (`spec.shapesBA :409`), slice updated L/R from `out` at offsets past `3nP` (`:640-642`) | medium |
| `apps/baselines/MainVitShampooTrain.lean` + lakefile `vit-tiny-shampoo-train` | clone the Muon trainer/exe | trivial |

**The host's baked-in assumption to break:** `Train.lean:601` `packed := (p.append m).append v`
and the output slices at `0/nP/2nP` (`:640-642`) assume three equal `nP`-sized blocks. Shampoo
appends two *unequal* blocks (`Σm²`, `Σn²` over 2D weights) → careful offset bookkeeping.

## 3. Risks / subtleties (where it'll fight you)

- **Init degeneracy + warmup.** Early `L≈0` ⇒ `L^{−1/4}` blows up ⇒ NaN at step 0. Init `L,R = εI`
  and/or warm up (AdamW for the first N steps, then engage the preconditioner — standard Shampoo).
- **NS inverse-root scaling.** Inverse-root NS needs the input scaled into its convergence basin
  (divide by `tr(L)`/`λ_max`); more iterations than Muon's single polar NS.
- **Memory.** +~21 MB L/R state for ViT-Tiny (`Σm²+Σn²` over the 72 block matrices). Fine.
- **Same LR caveat as Muon.** Shampoo's effective step scale differs from Adam's — don't reuse
  Adam's `3e-4` blindly (that's exactly what made the Muon A/B look bad before tuning).

## 4. Demo-first plan (per `math_threads.md §0.5`)

1. **Numerical jewel demo (cheap, do first).** A standalone Lean/Python script: (a) verify
   single-step Shampoo = Muon = `UVᵀ` numerically; (b) run accumulated-Shampoo vs Muon vs Adam on a
   toy quadratic / tiny MLP, plot trajectories. ~1 hr, zero plumbing. Validates the math AND the
   ε/warmup behavior you'll need before the integration.
2. **ViT A/B integration (the table in §2).** `vit-tiny-shampoo-train`, 3-way A/B
   (AdamW vs Muon vs Shampoo) on the 2×gfx1100 box — same harness as Muon
   (`run_vit_muon_ab.sh` template). **Sweep the Shampoo LR** (don't reuse Adam's), per the §3 caveat.
3. **Formalize later (the survivor).** Render-faithful `den=` tie reuses Muon's machinery; the
   NS-convergence deep cut for `λ^{−1/4}` is the same per-eigenvalue scalar lemma as Muon's `λ^{−1/2}`.

## 5. References

- Gupta, Koren, Singer, *Shampoo: Preconditioned Stochastic Tensor Optimization* (2018).
- Anil et al., *Scalable Second Order Optimization for Deep Learning* (2020) — the practical/EMA
  Shampoo + the inverse-root computation (coupled Newton, warmup, grafting).
- Martens & Grosse, K-FAC (2015) — the Fisher-side cousin.
- `planning/muon.md` (the single-step limit; the NS + diagonalization machinery this reuses),
  `planning/math_threads.md` Thread 1 (the ladder + the jewel derivation).
- In-repo anchors: `LeanMlir/MlirCodegen.lean` (`emitMuonUpdate` template, `emitTrainStepSig:6807`,
  dispatch `:6412`), `LeanMlir/Train.lean` (packing `:601`, slices `:640-642`, shapes `:409`),
  `ffi/iree_ffi.c:298-321` (the generic buffer-list FFI — why no C change is needed).
