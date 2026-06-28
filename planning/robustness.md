# Planning — Robustness: PGD attack vs Lipschitz certificate (Thread 4)

The robustness thread from `planning/math_threads.md` (Thread 4) — the dimension the repo lacks and
an external reviewer flags first. Two sides of adversarial robustness, bracketing the truth:

```
  certified robust acc   ≤   TRUE robust acc   ≤   PGD-empirical robust acc
   (proof, all attacks)                            (one strong attack)
```

- **PGD** = the attack = **upper bound** (empirical; finds adversarial examples; can't prove safety).
- **Lipschitz certificate** = the proof = **lower bound** (provable safe radius vs ALL attacks).

First target: **MNIST** (small net, fast, the certificate is computable exactly, it's the repo's
"hello world" verified net). Demo-first: cheap JAX attack now, formalize the certificate later.

## 1. PGD (the attack)

"Training, but the image is the variable and you climb the loss." Trained classifier `f`, image `x`,
label `y`, L∞ radius `ε`:

```
δ ← 0  (or uniform[−ε,ε])
repeat K:
    g = ∇_x L(f(x+δ), y)        # gradient w.r.t. the INPUT (your input-VJP, one layer earlier)
    δ ← δ + α·sign(g)            # L∞ step
    δ ← clip(δ, −ε, +ε)         # project to ε-ball
    x+δ ← clip(x+δ, 0, 1)       # valid pixels
adv_acc = fraction of (x+δ) still classified correctly
```

Standard MNIST (Madry et al.): L∞, **ε = 0.3** on [0,1] (the benchmark; sweep 0.1/0.2 for a curve),
**K = 40**, **α = 0.01**, random restart. The only new ingredient vs the repo is `∇_x` (input
gradient) — your VJP already computes the input cotangent internally (`mlpInputGrad_floatBridges`).

## 2. Lipschitz certificate (the proof)

For the MLP, computable exactly:
- `L = ∏ᵢ ‖Wᵢ‖₂` (ReLU is 1-Lipschitz; spectral norm of each weight via SVD).
- Per-example margin `m(x)` = top logit − runner-up.
- **Certified radius** `r(x) = m(x) / (√2·L)`: every `δ` with `‖δ‖₂ < r(x)` provably can't flip
  the prediction. Certified robust acc at `ε` = fraction with `r(x) ≥ ε`.

## 3. Implementation paths

| path | how | effort |
|---|---|---|
| **JAX phase-2 (DOING THIS first)** | a JAX MLP (matches the verified 784→512→512→10) + `jax.grad` w.r.t. input + the PGD loop + spectral-norm cert. ~50-100 lines, runs today | ~1-2 hrs |
| **In-system (phase-3 IREE)** | emit a `forward + input-gradient` kernel (new codegen output = the input cotangent the VJP has), PGD loop in the Lean host — the attack rides your PROVEN input-VJP | medium |
| **Formalize** | a `Lipschitz f L` predicate + per-layer spectral-norm bounds composing ⇒ a *verified* robust radius | larger |

## 4. Honest expected result (state up front)

Undefended MNIST MLP: clean ~98%, **PGD@ε=0.3 ≈ 0%** (undefended nets are catastrophically
vulnerable — that IS the headline), and the naive Lipschitz cert proves only a **tiny** safe radius
(the spectral-norm product is large). The demo shows a dramatic PGD-vs-certificate gap. Both true;
the gap is the research (Lipschitz-aware training, tighter composition). A loose-but-sound
certificate is still a real theorem — it under-promises.

## 5. Status / plan

- **JAX PGD demo — DONE 2026-06-28: `jax/demos/pgd_mnist.py` (runs on gfx1100, exit 0).** MLP
  784-512-512-10, clean test acc **97.83%** (matches the verified MLP). Results:
  - **L∞ PGD (Madry benchmark):** ε 0.1 → 6.25%, ε 0.2 → 0.05%, **ε 0.3 → 0.00%** (accurate net,
    catastrophically non-robust — the famous result reproduced).
  - **L2 sandwich:** layer ‖W‖₂ = [7.73, 8.97, 2.68] → **global L = 185.6**, median certified L2
    radius **0.051**. At L2 ε=0.5: PGD 77.45% (upper) vs certified 0.00% (lower) — a huge gap.
  - Takeaway: the attack crushes it; the naive Lipschitz cert is **sound but very loose** (the
    spectral-norm product is large ⇒ tiny certified radius). The gap is the research.
- **Phase-3 demo — DONE 2026-06-28: `mnist-linear-pgd` (verified net, real IREE pipeline).**
  `LeanMlir/VerifiedTrain.lean` `attackPgd` + `genLinearPgdStep` (the PGD-step kernel: forward →
  proven `dx=(softmax−onehot)·Wᵀ` input-VJP → L∞ sign-step → eps-ball project → [0,1] clip, all one
  IREE kernel; host iterates). Invoked via the generic `forwardF32` FFI (`onehot`+`x0` in the
  params blob, `nClasses:=d0`) — **no new FFI/C shim**. Attacks the ACTUAL verified net (vs the JAX
  demo's throwaway). Result: clean **92.10%**; L∞ PGD ε=0.1 → 24.02%, ε=0.2 → 0.35%, **ε=0.3 → 0.00%**.
  (Linear degrades more gracefully than the MLP at small ε — single small Lipschitz, smoother boundary
  — but still collapses by ε=0.3.) Run: `PATH=$PWD/.venv/bin:$PATH IREE_BACKEND=rocm .lake/build/bin/mnist-linear-pgd data`.
  Gotcha: `VerifiedTrain.compileVmfb` shells `iree-compile` from PATH (not the `.venv` lookup
  `Train.lean` uses) — put `.venv/bin` on PATH.
- **Certificate + L2 sandwich — DONE 2026-06-28 (in `attackPgd`).** `specNormW` (host power
  iteration on the small `WᵀW` Gram matrix) gives the **exact** logit-map Lipschitz `‖W‖₂ = 5.29`
  (linear = one layer, no product looseness); certified L2 radius `r(x)=margin/(√2·L)`; plus an L2
  PGD variant of `genLinearPgdStep` for an apples-to-apples bracket. Result (cert ≤ TRUE ≤ PGD):
  ε=0.5 **53.3% ≤ ? ≤ 79.4%**, ε=1.0 7.8% ≤ ? ≤ 52.3%, ε=1.5 0.5% ≤ ? ≤ 20.1%. The cert is
  **meaningfully tight** here (vs the JAX MLP's vacuous ~0% from `L=185.6`, a product of 3 norms) —
  the Thread-4 thesis shown empirically: tight cert ⟺ tight Lipschitz (one layer); the
  product-over-layers looseness is the thing to formalize-then-tighten.
- **Phase-3 MLP — DONE 2026-06-28: `mnist-mlp-pgd` (784→512→512→10).** `genMlpPgdStep` emits the
  proven `mlpInputGrad` VJP `dx=((g·W₂ᵀ⊙relu')·W₁ᵀ⊙relu')·W₀ᵀ` (ReLU masks via `compare GT`/`select`,
  the codegen idiom); training via the generic `mlpTrainStepV` FFI on the proof-rendered SGD
  `mlp_train_step` (6 weights in/out). Result: clean **97.8%**; L∞ PGD ε=0.1 → 5.8%, ε=0.2 → 0%.
  Cert: `‖W₀‖·‖W₁‖·‖W₂‖ = 3.16·3.64·3.41 = 39.2` (product) ⇒ **certified acc 0% even at L2 ε=0.5**
  (vacuous), vs L2 PGD 86%. **Linear vs MLP side-by-side (the Thread-4 thesis, empirical):**

  | net | clean | L∞ ε=0.1 | global L | cert@L2 0.5 | PGD@L2 0.5 |
  |---|---|---|---|---|---|
  | linear | 92.1% | 24.0% | 5.29 (exact) | **53.3%** | 79.4% |
  | mlp | 97.8% | 5.8% | 39.2 (product) | **0.0%** | 86.0% |

  More accurate ⇒ less robust; tight cert ⟺ tight Lipschitz (one layer) vs vacuous from the
  product-over-layers. The looseness to formalize-then-tighten is now concrete.
- **Phase-3 CNN — DONE 2026-06-28: `mnist-cnn-pgd` (the first conv rung,
  `planning/robustness_ladder.md`).** `genCnnPgdStep` emits the full input-VJP to `dx` —
  forward (saving every pre-act + the maxpool input) → softmax-CE seed → the proven backward,
  mirroring `verified_mlir/cnn_train_step.mlir`: dense adjoints + ReLU masks, **maxpool-back**
  (`select_and_scatter`, scatter the pooled cotangent to the argmax cells), and the two **conv
  input-VJPs** (transpose-`o,i` + spatial-`reverse` of the kernel, same padded conv) — **plus the
  final conv1 input-VJP the train step omits** (it only needs weight grads; the attack needs the
  pixel gradient). Trained via the generic `mlpTrainStepV` FFI on `cnn_train_step` (10 packed
  params); attack invoked through `forwardF32` (no new FFI). Conv-aware certificate:
  `specNormConvTapSum` — a **sound** conv operator-norm bound `‖T‖₂ ≤ Σ_{ky,kx} ‖W[:,:,ky,kx]‖₂`
  (each tap a `[out,in]` matrix; triangle inequality over norm-≤1 shifts) — times `specNormW` on
  the three denses. Result (10ep): clean **98.99%**; L∞ PGD ε=0.1 → 57.8%, ε=0.2 → 0.28%,
  **ε=0.3 → 0.00%** (catastrophic collapse at the Madry benchmark). Cert: conv1 Σtap=26.3 · conv2
  Σtap=9.53 · dense 2.21·2.84·2.03 → **global L = 3196** ⇒ **certified acc 0% at every L2 radius**
  (vacuous), vs L2 PGD ε=0.5 → 95.8%, ε=1.0 → 81.2%, ε=1.5 → 46.2%. **The depth-cliff, now with conv:**

  | net | clean | L∞ ε=0.1 | L∞ ε=0.3 | global L | cert@L2 0.5 | L2 PGD 0.5 |
  |---|---|---|---|---|---|---|
  | linear | 92.1% | 24.0% | 0.0% | **5.29** (exact) | **53.3%** | 79.4% |
  | mlp | 97.8% | 5.8% | 0.0% | **39.2** (3-layer product) | **0.0%** | 86.0% |
  | cnn | 99.0% | 57.8% | **0.0%** | **3196** (conv-aware product) | **0.0%** | 95.8% |

  linear-tight → MLP-vacuous → CNN-more-vacuous: even *exact* per-layer conv norms wouldn't help —
  the looseness is the **product**, not the per-tap estimate. (The CNN's higher L∞ ε=0.1 number is
  the smoother conv loss surface; it still collapses to exactly 0% by ε=0.3.) Run:
  `PATH=$PWD/.venv/bin:$PATH IREE_BACKEND=rocm .lake/build/bin/mnist-cnn-pgd data`
  (`CNN_PGD_EPOCHS=1` for a quick smoke). Log: `runs/pgd_cnn_phase3.log`.
- **Lipschitz-margin certificate FORMALIZED — DONE 2026-06-28: `LeanMlir/Proofs/LipschitzCert.lean`
  (3-axiom clean).** The cert is no longer just a number — it's a **theorem**.
  `lipschitz_margin_certified_radius` (Tsuzuku et al. 2018): if the logit map `f` is `L`-Lipschitz
  in L2 and class `i` leads every other by margin `m` at `x`, then **every `δ` with
  `‖δ‖₂ < m/(√2·L)` leaves `i` the argmax** — a provable safe radius vs *all* attacks (the lower
  bound of `cert ≤ TRUE ≤ PGD`, where PGD only upper-bounds via one attack). Engine:
  `logit_gap_stable` (a pairwise logit gap is `(√2·L)`-Lipschitz) built on `coord_pair_bound`
  (`(vᵢ−vⱼ)² ≤ 2‖v‖²` — the `√2` is `‖eᵢ−eⱼ‖₂`). `LipschitzL2.comp` (constants multiply) +
  `clm_lipschitzL2` (a linear layer is `‖A‖`-Lipschitz, `‖A‖` = the spectral norm `specNormW`
  estimates) make the per-layer **product** `L = ∏ᵢ‖Wᵢ‖₂` a *sound* global constant — proving
  exactly *why* the product is valid (and, past one layer, loose: the depth-cliff the demos show).
  Stated over `EuclideanSpace ℝ (Fin k)` (genuine L2 + the `√2`), domain any normed space. Audited
  in `tests/AuditAxioms.lean`.
- **Spectral-norm training — DONE 2026-06-28: `mnist-mlp-spectral` (the gap-shrinking lever).**
  `attackPgdSpectralMlp` trains the MLP with **projected SGD onto the spectral ball** — every 20
  proof-rendered steps (+ once at the end) each weight `Wᵢ` is rescaled to `‖Wᵢ‖₂ ≤ c`
  (`projectSpectral` via the matrix-free `specNormMV` + `F32.scaleShift`; the verified CE gradient
  stays in the proven kernel, projection is host-side weight rescaling only) — capping `L ≤ c³`.
  Sweeping `c` runs the **empirical face of `lipschitz_margin_certified_radius`** (smaller `L` ⇒
  bigger certified radius `m/(√2·L)`). Result (12ep, `runs/spectral_mlp_phase3.log`):

  | cap c | clean% | global L | cert@L2 0.5 | L2 PGD 0.5 | L∞ PGD 0.1 |
  |---|---|---|---|---|---|
  | ∞ (none) | 97.76 | 39.21 | **0.00%** | 86.0 | 5.8 |
  | 3.0 | 97.81 | 27.89 | 0.00% | 87.8 | 8.6 |
  | 2.0 | 96.19 | 8.38 | **24.16%** | 89.7 | **53.0** |
  | 1.5 | 88.84 | 3.38 | **42.30%** | 79.0 | 49.9 |
  | 1.0 | 64.38 | 1.02 | 27.12% | 54.2 | 38.3 |

  The baseline reproduces the undefended MLP (97.8%, L=39.2, cert 0%). **The headline: at c=2.0 the
  vacuous product cert goes 0% → 24.2% *and* L∞ PGD robustness jumps 5.8% → 53% (9×), for only 1.6%
  clean accuracy.** The cert↔PGD gap at L2 0.5 shrinks 86 → 37 pts (c=1.5). It's a genuine
  trade-off curve, non-monotone: c=1.5 maximizes certified acc (42.3%), and c=1.0 *over*-constrains
  (cert falls to 27% as margins shrink faster than `L`, clean tanks to 64%) — there's an optimal `c`.
  Run: `PATH=$PWD/.venv/bin:$PATH IREE_BACKEND=rocm .lake/build/bin/mnist-mlp-spectral data`
  (`SPECTRAL_EPOCHS=2` for a smoke).
- **Spectral-norm training, CNN — DONE 2026-06-28: `mnist-cnn-spectral`.** `attackPgdSpectralCnn`
  pushes the same projected-SGD lever to the conv net: `projectSpectral` now caps **both** the dense
  `‖Wᵢ‖₂` and the **conv tap-sum bound** (`specNormConvTapSum`, the same quantity the CNN cert uses,
  scaling the whole kernel) at `c`. Result (10ep, `runs/spectral_cnn_phase3.log`):

  | cap c | clean% | global L | cert@L2 0.1 | cert@L2 0.25 | cert@L2 0.5 | L∞ PGD 0.1 |
  |---|---|---|---|---|---|---|
  | ∞ (none) | 98.99 | 3196 | 0.0% | 0.0% | 0.0% | 57.8 |
  | 2.0 | **96.96** | 34.1 | **53.7%** | 0.0% | 0.0% | 53.5 |
  | 1.5 | 90.35 | 8.06 | **68.7%** | 17.3% | 0.0% | 70.5 |
  | 1.2 | 65.41 | 2.60 | 46.9% | 21.3% | 0.01% | 47.2 |
  | 1.0 | 28.92 | 1.01 | 21.5% | 13.0% | 1.98% | 20.7 |

  Baseline reproduces the verified CNN exactly (98.99%, L=3196). **The CNN certifies — but only at a
  *smaller* radius than the MLP:** at c=2.0 the cert goes 0% → 53.7% **@ L2 ε=0.1** for −2% clean,
  yet reaching cert@L2 0.5 needs collapsing to 29% clean (c=1.0). Why it's harder than the MLP: a
  **5-layer** product (`L ≤ c⁵`) and the conv tap-sum is *loose* (by ≈√9/conv), so projecting onto
  it over-penalizes the convolutions. This is the case where the **exact (Sedghi–Gupta–Long FFT)
  conv spectral norm** would actually pay — the per-layer estimate, not just the product, is now the
  bottleneck. Run: `… .lake/build/bin/mnist-cnn-spectral data` (`SPECTRAL_EPOCHS=2` for a smoke).
- **CIFAR rung — DONE 2026-06-28: `cifar-pgd` (the deeper conv rung).** The CNN attack + spectral
  drivers were refactored into generic `attackPgdConvNet`/`attackPgdSpectralConvNet` (kernel-gen as a
  parameter) and gained **best-θ checkpointing** (plain SGD on the deeper CIFAR net diverges late —
  it peaked 70% at epoch 8 then went NaN; attacking the best checkpoint keeps the demo + cert finite).
  `genCifarPgdStep` (4 conv input-VJPs + 2 maxpool-backs + conv1-VJP, `runs/pgd_cifar_phase3.log`):
  clean **70.0%** → L∞ PGD ε=0.1/0.2/0.3 → **0%/0%/0%**; conv-aware **global L = 942,040** (7-layer
  product) ⇒ cert 0% everywhere; L2 PGD ε=0.5 → 8.8%. The depth-cliff, four rungs:
  **linear 5.3 → MLP 39 → MNIST-CNN 3,196 → CIFAR-CNN 942K.** The spectral-CIFAR study
  (`cifar-spectral`) confirmed the limit: caps c=6/4/3 train (45–55%) but L stays vacuous (≥1357),
  c≤2 kills the 7-layer net — **no cap makes CIFAR's cert non-vacuous** with the tap-sum bound. Run:
  `… .lake/build/bin/cifar-pgd data` (`CIFAR_PGD_EPOCHS=1` for a smoke).
- Next: tighter-than-product composition (the actual research lever — the formalization now makes
  "tighten the bound" a concrete theorem-strengthening target), and the exact-FFT conv spectral norm
  for the CNN (the per-layer bottleneck the spectral-CNN study exposed). CIFAR is the same attack
  recipe + BN folding; Imagenette's real cert is randomized smoothing, not the product (vacuous).

## 6. References

- Madry et al., *Towards Deep Learning Models Resistant to Adversarial Attacks* (PGD, 2017).
- Szegedy et al. (2013) / Goodfellow et al. (FGSM, 2014) — adversarial examples / the sign-gradient.
- Tsuzuku, Sato, Sugiyama, *Lipschitz-Margin Training* (2018) — the margin/Lipschitz certified radius.
- Sedghi, Gupta, Long (2019) — exact conv spectral norms (for the conv version later).
- `planning/math_threads.md` Thread 4; in-repo `mlpInputGrad_floatBridges` (the input-VJP the
  in-system attack would reuse), `apps/mnist/` (the verified MNIST nets).
