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
- Next: add the certificate to the phase-3 driver (linear: a single `‖W‖₂` via power iteration);
  climb to the MLP/CNN input-grad kernels; then the `Lipschitz f L` formalization (makes the
  certificate a theorem). To shrink the PGD↔cert gap: Lipschitz-margin training / tighter composition.

## 6. References

- Madry et al., *Towards Deep Learning Models Resistant to Adversarial Attacks* (PGD, 2017).
- Szegedy et al. (2013) / Goodfellow et al. (FGSM, 2014) — adversarial examples / the sign-gradient.
- Tsuzuku, Sato, Sugiyama, *Lipschitz-Margin Training* (2018) — the margin/Lipschitz certified radius.
- Sedghi, Gupta, Long (2019) — exact conv spectral norms (for the conv version later).
- `planning/math_threads.md` Thread 4; in-repo `mlpInputGrad_floatBridges` (the input-VJP the
  in-system attack would reuse), `apps/mnist/` (the verified MNIST nets).
