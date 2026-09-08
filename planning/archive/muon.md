# Planning — Muon: a polar-projection optimizer for the transformer chapters

Notes + a design sketch for adding **Muon** (MomentUm Orthogonalized by Newton–Schulz)
to the book/codebase. Muon is *not* a CIFAR demo — it's the **capstone of the optimizer
arc**, lives in the **ViT chapter**, and is foreshadowed (not explained) at the end of
the CIFAR optimizer ablation. This doc captures the math, where it fits pedagogically,
what it touches in the codegen, and exactly how far the proofs can reach.

## 0. TL;DR

- **What it is.** Muon replaces the per-parameter update with a **polar projection** of
  the (momentum) gradient onto the (semi-)orthogonal matrices: take `G = UΣVᵀ`, keep the
  directions `UVᵀ`, throw away the magnitudes `Σ`. Every direction the gradient pointed in
  gets an equal-size step ⇒ the update uses the full rank of the weight matrix each step.
- **Why a matmul iteration, not an SVD.** SVD is a sequential factorization GPUs hate.
  Newton–Schulz computes `UVᵀ` *approximately* with ~5 pure-GEMM iterations — the same
  primitive the forward/backward already runs on. Same operator, cheap GPU-native route.
- **Where it goes.** ViT chapter (its habitat — ViT-Tiny is ~all 2D matrices). Hybrid
  with AdamW: **Muon on the 2D weights, AdamW on embeddings / CLS / pos-embed / LN gains /
  head / biases.** CIFAR chapter only plants the seed.
- **What we can prove.** **Render-faithfulness is reachable** (the NS update unrolls to a
  matmul chain; reuses the existing dense/matmul VJP vocabulary — a `den=` tie like the
  others). **Descent is open** (Muon's guarantee is spectral-norm steepest descent, a
  different animal from the `lr·‖∇‖²/2` Frobenius-MVT proofs in `SgdDescent*`). An optional
  **verified-NS-convergence** deep cut reduces to a *1D* real-analysis fact.
- **The closure for the book.** The property that makes Muon **fast at scale** (it's all
  matmul) is the *same* property that makes it **clean to verify** (no new primitive). Fast
  and provable are one coin here — a good note to end the optimizer thread on.

## 1. The math (one mental model)

The whole thing falls out of "a weight is a **linear map**, not a bag of scalars."

SGD/momentum/Adam are *elementwise*: flatten `W`, update each scalar on its own (Adam =
per-coordinate RMS rescaling — diagonal preconditioning in the standard coordinate basis).
They never see the matrix structure.

Muon works in the gradient's **singular basis**. Write the (momentum) gradient by its SVD:

```
G = U Σ Vᵀ           directions (U,V)  +  per-direction magnitudes (Σ)
```

The raw `G` is wildly anisotropic — a few singular values dominate, so `−lr·G` collapses
onto 2–3 directions. Muon flattens the spectrum:

```
update = U Vᵀ         set every singular value to 1; keep directions; step equally
W ← W − lr · (U Vᵀ)
```

**This is a polar projection.** `G = QP` with `Q = UVᵀ` (orthogonal/partial-isometry) and
`P = VΣVᵀ` (PSD). `Q` is the **nearest orthogonal matrix to G** (orthogonal Procrustes):

```
Q = U Vᵀ = argmin_{O orthogonal} ‖G − O‖_F
```

So Muon = "momentum → project onto the orthogonal manifold → step." Three equivalent views:

| hat | reading |
|---|---|
| linear algebra | nearest orthogonal matrix (Procrustes); flatten the singular spectrum to 1 |
| optimization | steepest descent under the **spectral** norm (dual view of the same `UVᵀ`) |
| QM | polar projection onto the unitaries — `A = U\|A\|`, keep `U`, drop `\|A\|` |

Caveat on the word "projection": it's a **nonlinear nearest-point** projection onto the
Stiefel/orthogonal manifold, not an idempotent linear projector. It *is* idempotent as a
retraction (`polar(polar(G)) = polar(G)`).

### Newton–Schulz: the cheap polar projection

Don't compute the SVD — it's a sequential factorization (bidiagonalize + iterative QR),
data-dependent control flow, bandwidth-bound, fp32/fp64 for stability; it stalls the GPU.
Newton–Schulz produces `UVᵀ` with **only matmuls**: normalize `G` so all `σ ∈ (0,1]`, then
iterate a tuned odd quintic ~5×:

```
X ← a·X + b·(XXᵀ)X + c·(XXᵀ)²X        (a,b,c) ≈ (3.4445, −4.7750, 2.0315)
```

The key fact (and why it's clean to verify): the iteration **touches only the singular
values, never the directions**. Since `X = UΣVᵀ`,

```
(XXᵀ)X = U Σ³ Vᵀ,   (XXᵀ)²X = U Σ⁵ Vᵀ   ⟹   p(X) = U (aΣ + bΣ³ + cΣ⁵) Vᵀ
```

so it's secretly the *scalar* map `σ ↦ aσ + bσ³ + cσ⁵`, applied per singular value, with
`U,V` carried along. Coefficients tuned so that map drives `(0,1] → ≈1` in ~5 steps (lands
in a band near 1 — rough is fine; we recompute next step anyway).

This is the **matrix-function-iteration** family (Higham, matrix sign / polar): the classic
inverse-free polar iteration is the cubic `X ← ½X(3I − XᵀX)`; Muon's quintic just converges
faster. Muon's contribution is realizing approximate polar projection is the *right object*
for an optimizer update.

### Intuition bridges (use these in the prose)

- **Graphics / animation (Procrustes & polar).** `UVᵀ` is exactly **Kabsch** (best-fit
  rotation aligning two point sets) and the **polar-from-deformation-gradient** trick
  (`F = RS`, keep rotation `R`, drop stretch `S`) from co-rotational FEM, shape matching,
  ARAP, and re-orthogonalizing drifted bone matrices. Muon = the same nearest-rotation
  projection, on a 768×768 weight-gradient instead of a 3×3 pose. Graphics already dodges
  SVD with iterative/Newton polar in real-time skinning — the *same* perf reason as NS.
  One-liner for vision/graphics readers: *"Muon is the re-orthogonalization you do in
  skinning, scaled up to weight matrices."* ARAP's "compute best rotation, apply, repeat"
  is even the right loop shape.
- **Normalization-as-axis.** BN normalizes across the batch axis, LN across the feature
  axis, Muon across the singular-value axis of the *update*. Same "kill the anisotropy"
  instinct, three different axes — a unifying lens for the back half of the book.

## 2. Where it fits the book

CIFAR = the **primitives lab** (small, fast, ablatable): teach SGD → momentum → Adam and
plant the *two axes* — the optimizer itself **and** the schedule/budget ("longer convergence
schedules"). Name the shared assumption — *all three treat a weight as independent scalars*
— because that's the assumption Muon breaks.

The arc the rest of the book walks:

```
elementwise menu   SGD → momentum → Adam(W)      "rescale per coordinate"
       │
schedule axis      warmup · cosine · long runs   "the budget co-designs the curve"
       │
conv → transformer why Adam, why 2D matrices      the pivot
       │
geometry frontier  MUON / spectral methods        "treat the weight as a map"   ← capstone
```

Muon must be **last**: it's scaffolded on (a) Adam-is-still-per-coordinate, (b) transformers
are stacks of 2D matrices, (c) the speedrun "convergence-rate-is-everything" mindset that
*produced* Muon (nanoGPT speedrun). That last point ties it directly to the book's stated
"optimizers × longer schedules" theme: Muon is the optimizer the convergence-race produced.

**CIFAR foreshadow sentence (plant, don't explain):**

> *Every optimizer in this chapter treats a weight as a pile of independent numbers, and
> tunes each on its own. Hold onto that assumption. Once our networks become stacks of large
> weight **matrices** — the transformers ahead — we'll meet optimizers that instead treat
> each weight as a linear map and move all of its directions as one. That's where convergence
> rate and verifiability both get interesting again, and it's where this book is headed.*

**ViT-chapter section skeleton:** motivate from ViT's matrices (count them) → polar/SVD
intuition (QM + Procrustes/graphics bridge) → Newton–Schulz as the matmul approximation →
the Muon/AdamW param partition (below) → render-faithful tie (what we prove) + the descent
gap (what we don't) → optional verified-NS-convergence box → a compute-matched
Muon-vs-Adam curve to cash out the schedule theme.

### The param partition (a teaching artifact + the actual config)

Muon is a hybrid, not a drop-in. Split ViT-Tiny's parameter inventory:

| → **Muon** (2D weight matrices) | → **AdamW** (everything else) |
|---|---|
| Q, K, V, O projections (×12 blocks) | patch-embed conv, CLS token, pos-embed |
| MLP `fc1`, `fc2` (×12 blocks) | every LayerNorm γ/β, classifier head, all biases |

This table reuses the exact param list the chapter already has, and makes "Muon for the
matrix guts, AdamW for the edges" tangible.

## 3. What it touches in the codegen

- **The gradient is unchanged.** Muon is *only* an optimizer — it consumes the same
  `∂Loss/∂W` the ViT VJP+tie apparatus already certifies. The entire proof machinery for the
  chapter is reused verbatim; Muon swaps the *function that turns the certified gradient into
  a step*. Strong reassuring message.
- **New op = a straight-line matmul program.** Add a Muon update emitter: momentum buffer
  (already exists for SGD-momentum), the initial `G`-normalization, then ~5 unrolled NS steps
  (`XXᵀ`, `(XXᵀ)X`, scalar-mul, add). Every op is one the *attention* codegen already emits
  for matmuls — a new **composition**, not a new primitive. Unrolled ⇒ a finite expression
  in `G` (no loop / fixpoint in the IR).
- **Extension points.** `OptimizerKind` enum gets a `.muon` variant (cf. how RMSprop landed);
  the renderer (`ViTRender.lean`, which already has an AdamW tail) gets a Muon tail for the
  2D-weight params and keeps the AdamW tail for the rest. A new `SHlo` node (e.g.
  `SHlo.muonUpdate`) or a small composite of existing matmul nodes carries the NS chain.
- **Host vs device.** NS is GPU-native (bf16 GEMM), so it belongs *in* the StableHLO, unlike
  the fp8 host-quant emulation (`E4M3Quant.lean`). No FFI changes.

## 4. The verification story (how far the proofs reach)

| claim | reachable? | route |
|---|---|---|
| **render-faithful**: `den(emitted Muon update) = momentum-then-NS-orthogonalize(G)` | ✅ tractable | unroll NS → matmul chain; `den=` tie like `vit_net_tied_certified`; reuses `dense_has_vjp`/matmul vocabulary; no new proof tech |
| **NS convergence**: the iterate actually → `UVᵀ` | ⚠️ optional deep cut | diagonalize ⇒ the matrix iteration ⟺ the **1D** quintic `σ↦aσ+bσ³+cσ⁵` on `(0,1]`; a real-analysis fixed-point fact (Mathlib `Analysis` + the margin machinery) |
| **descent**: the step decreases the loss | ❌ open | needs spectral-norm steepest-descent / modular-norm theory (Bernstein–Newhouse / Jordan); the `SgdDescent*` Frobenius-`‖∇‖²` MVT argument does **not** transfer |

**Honest framing (same ladder as the rest of the book):** we certify the optimizer *computes
what we said it computes* (render-faithful). We do **not** claim the result equals the exact
SVD `UVᵀ` (NS is approximate — that's the optional deep cut) or that it decreases the loss
(open, different theory). Muon lands exactly at the frontier where the optimizer-sophistication
axis and the proof-difficulty axis both top out — the right close for the chapter that ends
Part I.

**The verified-NS-convergence deep cut is very on-brand.** "Does this scary matrix iteration
converge?" collapses to "does one quintic on `(0,1]` converge to 1?" by the exact
diagonalize-into-scalars move that makes the orthogonalization work in the first place. It'd
be a genuinely novel *verified Newton–Schulz convergence* result. Caveat: the standard Jordan
coefficients are empirically tuned, not proven-optimal, and the iteration isn't a clean
monotone contraction everywhere — the proof is about the *specific* quintic on the relevant
band, tractable but not a 3-liner. Put it in a "going deeper" box, optional.

## 5. Step order (when we build it)

1. **Demo first (unverified path). ✅ DONE 2026-06-27 (uncommitted).** Added `.muon` to
   `OptimizerKind` + `TrainConfig.useMuon`; `emitMuonUpdate` emits the 5-step NS chain as pure
   `dot_general`+scalar StableHLO (`X·Xᵀ` via `contracting_dims=[1]x[1]`, no transpose; momentum
   in m-slot, v-slot passthrough → arity unchanged); dispatch routes 2D weights (both dims ≥16)
   → Muon, rest → AdamW (= the canonical partition: 72 ViT block matrices Muon, patch-conv/LN/
   biases/[192,10] head AdamW). Exe `vit-tiny-muon-train` (`apps/baselines/MainVitMuonTrain.lean`,
   config sets `useAdam:=true`+`useMuon:=true` so the host drives the Adam m+v state path — the
   Muon module is Adam-signature-identical). VALIDATED end-to-end on gfx1100: full ViT train-step
   `iree-compile`s, trains, step-0 loss=3.35 finite/sane, ~444ms/step. A/B launcher
   `run_vit_muon_ab.sh` (Muon GPU0 vs AdamW GPU1). Run the compute-matched curve next.
2. **Render-faithful tie.** Build the proof-tied Muon tail in `ViTRender.lean`; prove the
   `den=` tie for the 2D-weight params (the NS chain denotes the explicit quintic-in-`G`),
   AdamW tail unchanged for the rest. Audit to 3-axiom closure (`tests/AuditAxioms.lean`).
3. **(Optional) verified-NS-convergence box.** The 1D quintic fixed-point lemma.
4. **Prose.** CIFAR foreshadow sentence; ViT section per the skeleton in §2.

## 6. Open questions / risks

- **Coefficients & step count.** Pin the exact `(a,b,c)` and iteration count we ship; the
  empirical band-near-1 behavior is what the render-tie certifies *denotes*, independent of
  whether it's the optimal tuning.
- **Conv weights?** If we ever want Muon outside transformers, conv kernels reshape to
  `out_ch × (in_ch·kh·kw)` — works but murkier; out of scope for the ViT-chapter framing.
- **Momentum/AdamW interaction.** The hybrid means two optimizer states coexist; make sure the
  param-partition routing is faithful in both the trainer and the render.
- **bf16 NS stability.** NS in bf16 is standard but worth a numerical sanity check in step 1.
- **"1:1 with the book."** Per the Results-table decision (see `verified-path-table-rerun`),
  any Muon numbers are additive to the optimizer story, not a change to the headline net table.

## 7. References

- Jordan, *Muon: An optimizer for the hidden layers of neural networks* (2024) — the NS
  coefficients, the speedrun context.
- Bernstein & Newhouse, *Modular norms / Old optimizer, new norm* — the spectral-norm
  steepest-descent framing (why `UVᵀ` is the natural step).
- Higham, *Functions of Matrices* — the matrix sign / polar Newton–Schulz family.
- Orthogonal Procrustes / Kabsch; polar decomposition in co-rotational FEM, shape matching,
  ARAP (the graphics intuition bridge).
- In-repo: `LeanMlir/Proofs/SgdDescent*.lean` (the descent proofs Muon does *not* fit),
  `ViTRender.lean` (AdamW tail precedent), `ViTTiePoC.lean` (`vit_net_tied_certified`, the
  render-tie template), `E4M3Quant.lean` (host-quant precedent — contrast: NS stays on-device).

## 8. Math (chapter appendix)

The mathematical skeleton for the ViT-chapter section. Five pieces; the two load-bearing
lemmas (§8.2, §8.4) are short enough to prove in place.

### 8.1 The master equation — every optimizer is steepest descent under *some* norm

For loss `L`, weight `W`, gradient `G = ∇_W L`, model the step as a trust-region quadratic:

```
ΔW = argmin_Δ  ⟨G, Δ⟩ + (1/2η)·‖Δ‖² .
```

The choice of norm ‖·‖ IS the optimizer's identity:

| optimizer | norm on Δ | step ΔW |
|---|---|---|
| SGD  | Frobenius ‖Δ‖_F | −η·G |
| Adam | per-coordinate Σ Δ²_ij / v̂_ij | −η·G ⊘ (√v̂ + ε) |
| **Muon** | **spectral** ‖Δ‖₂ (largest singular value) | **−η·UVᵀ** |

SGD measures the step democratically over entries; Adam per-entry (diagonal metric); Muon by
*operator gain*. That one substitution is the whole method.

### 8.2 Why the spectral norm gives UVᵀ (dual-norm derivation)

Steepest-descent direction under ‖·‖ = the unit-norm Δ best aligned with −G, i.e.
`argmax_{‖Δ‖₂ ≤ 1} ⟨G, Δ⟩`. With the thin SVD `G = U Σ Vᵀ`:

```
⟨G, UVᵀ⟩ = tr(Gᵀ U Vᵀ) = tr(V Σ Uᵀ U Vᵀ) = tr(Σ) = Σ σ_i = ‖G‖_{S₁}   (nuclear norm).
```

Since ‖UVᵀ‖₂ = 1 (all its singular values are 1) and Hölder for Schatten norms gives
`⟨G,Δ⟩ ≤ ‖G‖_{S₁}·‖Δ‖₂`, the max is attained exactly at Δ = UVᵀ. ∎
→ spectral-norm steepest descent = step along UVᵀ. **Flatten the spectrum to 1, keep the directions.**

### 8.3 That direction is the polar factor / nearest rotation

UVᵀ is the **orthogonal Procrustes** solution — the nearest semi-orthogonal matrix:

```
UVᵀ = argmin_{O: OᵀO = I} ‖G − O‖_F = Q   in the polar decomposition  G = Q P,  P = V Σ Vᵀ ⪰ 0.
Closed form:  Q = G (GᵀG)^(−1/2) = U Σ Vᵀ · V Σ⁻¹ Vᵀ = U Vᵀ.
```

Same Q as Kabsch alignment and co-rotational-FEM rotation-from-deformation-gradient. The
`(GᵀG)^(−1/2)` form is the bridge to computing the inverse square root WITHOUT an SVD/inverse.

### 8.4 Newton–Schulz — the diagonalization lemma

For `X = U Σ Vᵀ` and any **odd** polynomial `p(x) = a x + b x³ + c x⁵`, define
`p_mat(X) := aX + b(XXᵀ)X + c(XXᵀ)²X`. Then since `XXᵀ = U Σ² Uᵀ`:

```
(XXᵀ)X = U Σ³ Vᵀ,   (XXᵀ)²X = U Σ⁵ Vᵀ
⇒   p_mat(X) = U · p(Σ) · Vᵀ ,   p(Σ) = diag(p(σ_i)).
```

**The matrix iteration acts independently on each singular value, leaving U,V untouched** — a
matrix recurrence collapses to the scalar system `σ_{k+1} = p(σ_k)`. ∎

Setup: normalize `X₀ = G/‖G‖_F`; since `‖G‖_F = √(Σσ²) ≥ σ_max = ‖G‖₂`, all σ ∈ (0,1].
Goal: drive σ_k → 1 (then X_k → UVᵀ). Two regimes to contrast:

- **Cubic (textbook, provably convergent):** `X ← ½X(3I − XᵀX)`, i.e. `φ(σ)=1.5σ−0.5σ³`.
  Fixed point φ(1)=1, φ'(1)=0 ⇒ **quadratic convergence** to 1 on (0,√3). The version to PROVE.
- **Quintic (Jordan's Muon, tuned for speed):** (a,b,c)=(3.4445,−4.7750,2.0315). Large φ'(0)=a
  escapes small σ fast, but φ(1)=0.701 ≠ 1 — does NOT fix 1; parks the spectrum in a *band*
  around 1 after ~5 steps. Approximate orthogonalization traded for speed (fine — recompute next step).

That split is both a teaching beat and the answer to "which iteration do we formalize" (the cubic).

### 8.5 The Muon update (algorithm) and the shape-scale

For a 2D weight `W ∈ ℝ^{m×n}`, gradient `G_t`:

```
B_t = μ B_{t-1} + G_t                                   (heavy-ball momentum)
O_t = NS₅( B_t / (‖B_t‖_F + ε) )  ≈  U_{B_t} V_{B_t}ᵀ    (approx polar factor)
W  ← W − η·s·O_t ,   s = √(max(1, m/n))                  (RMS shape-match)
```

Why s: after orthogonalization `‖O_t‖_F = √rank ≈ √min(m,n)`, so per-element update RMS depends
on shape; s rescales it comparable across layers (and to Adam's ≈unit-RMS update) — why Muon
transfers LRs across a transformer's differently-shaped weights. (Conventions vary; s is a tunable.)

### 8.6 Verification math — what's provable, as theorems (the book's spine)

- **Render-faithful (reachable).** Each NS step is `dot_general`s, so by §8.4 the emitted program
  denotes the closed form `den(NS₅)(B) = U·φ^∘5(Σ)·Vᵀ`. Theorem to state: `den(emitted Muon step)`
  = the §8.5 symbolic update — a `den=` tie like the chapter's others, no new proof tech.
- **Convergence (optional deep cut).** By §8.4 reduces to a **1-D** claim: `φ(σ)=1.5σ−0.5σ³` is a
  contraction toward 1 on (0,√3) — a Mathlib real-analysis fixed-point lemma. (Prove the *cubic*.)
- **Descent (open, by design).** The Frobenius bound `L(W−ηG) ≤ L(W) − η‖∇‖_F²(1−ηβ/2)` does NOT
  transfer. The spectral analogue, under spectral-norm smoothness β₂:
  `L(W − η UVᵀ) ≤ L(W) − η‖∇L‖_{S₁} + η²β₂/2` — descent measured in the **nuclear** norm, a
  different (unformalized) argument. The honest frontier.

### 8.7 Empirical anchor (A/B, fill on run completion)

ViT-Tiny / Imagenette, identical recipe (AdamW cosine+warmup, LS 0.1, wd 1e-4, 80ep, bs 32),
`vit-tiny-muon-train` (Muon-on-2D + AdamW-edges) vs `vit-tiny-train` (AdamW). ~122 s/epoch/gfx1100,
~2.75 h for 80ep. **HONEST RESULT (full 80ep, 2026-06-28): AdamW wins.**
- **Final val acc (ep80): AdamW 70.1% (2737/3904) vs Muon 66.7% (2605/3904)** — ~3.4 pts; gap
  opened by the 2nd eval (~10ep) and held. AdamW peaked 70.5% @ ep70; Muon climbed monotonically
  but plateaued ~3.4 pts below.
- Final train loss: AdamW 0.523 vs Muon 0.544 (AdamW also lower).
- Muon led on *training loss* mid-run (crossover ~ep6, +0.10 by ep14) but it **reversed in cosine
  decay** and never translated to val — the "faster train loss ≠ better generalization" caveat, live.

**The cause is almost certainly mistuning, not Muon.** The demo gave Muon **Adam's LR (3e-4) and
recipe verbatim**; Muon's orthogonalized updates are on a different scale (unit singular values ×
shape factor) and need their OWN, typically much larger LR (~1e-2–5e-2). Plus Imagenette/ViT-Tiny is
small — not Muon's scale regime. A single shared-LR point is not a fair Muon eval. **Next: a Muon LR
sweep** (the real ablation), before any claim. This is the demo-first protocol working — it surfaced
the mistuning before a line of proof was written.
