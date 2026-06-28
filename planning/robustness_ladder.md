# Planning — Robustness ladder: MNIST-CNN → CIFAR → Imagenette?

Scoping how far to chase the phase-3 robustness demo (`planning/robustness.md`: PGD attack via the
proven input-VJP kernel + Lipschitz certificate) up the architecture ladder. Done so far:
**linear** (exact cert, certifies 53% @ L2 0.5) and **MLP** (product cert, vacuous 0%). The key
realization that shapes everything below: **there are TWO ladders, and they diverge.**

## 0. TL;DR — the two ladders diverge

- **The ATTACK ladder scales** (mechanically). PGD just needs `∇_x loss`, which is the proven
  whole-net backward your codegen already computes — thread it to `dx`, emit a kernel, iterate.
  MNIST-CNN → CIFAR → Imagenette are all *possible*; cost (kernel size, compile time, PGD wall-clock)
  is the only thing that grows.
- **The naive-Lipschitz CERT ladder does NOT scale.** `L = ∏ᵢ ‖Wᵢ‖₂` is meaningful at **1 layer**
  (linear: exact, tight) and **already vacuous at 3** (MLP: L=39, certifies nothing). Past that the
  product is astronomically loose **by construction** — deeper nets only reconfirm vacuity. So the
  cert *story* is essentially complete at linear-vs-MLP.
- **Verdict on Imagenette:** the **attack** is possible but expensive (1–2 MB kernels, ~10–15 min
  iree-compile each, slow PGD) and only reconfirms "deep nets are fragile." The naive **cert** is
  pointless there. The honest deep-net certificate is a *different paradigm* — **randomized
  smoothing** (Cohen et al. 2019), which is architecture-agnostic, doesn't degrade with depth, and
  is itself a clean runnable demo. That, not the Lipschitz product, is the real Imagenette robustness story.

## 1. The attack ladder (reuse the proven backward → `dx`)

Each rung = a `gen<Net>PgdStep` kernel (forward, saving the pre-acts the backward needs → the
proven input-VJP → L∞/L2 step+project) + an `attackPgd<Net>` driver (train on the proof-rendered
step, attack). Pattern established by `mnist-linear-pgd` / `mnist-mlp-pgd`.

| rung | new kernel ops vs MLP | cert (per-layer norm) | effort | notes |
|---|---|---|---|---|
| **MNIST-CNN** ✅ **DONE 2026-06-28** (`mnist-cnn-pgd`) | conv input-VJP (transpose-`o,i`+spatial-`reverse` kernel); **maxpool-back** (`select_and_scatter`); reshape glue — all mirrored from `cnn_train_step.mlir` + the final conv1 input-VJP it omits | conv-aware **product**: `specNormConvTapSum` (sound `Σ_tap‖W[:,:,ky,kx]‖₂`) × `specNormW` denses → **L = 3196** | **medium** (done) | clean **99.0%**, L∞ ε=0.3 → **0.0%** (collapse), cert 0% everywhere (vacuous). Depth-cliff now visual: linear L=5.3 → mlp L=39 → cnn L=3196. `runs/pgd_cnn_phase3.log` |
| **CIFAR** ✅ **DONE 2026-06-28** (`cifar-pgd` + `cifar-bn-pgd`) | `genCifarPgdStep`: 4 conv input-VJPs + **2** maxpool-backs + conv1-VJP. `genCifarBnPgdStep`: + the **BN grad-input 3-term formula** ×4 — `cifar_bn`'s BN is *instance* norm (per-image), so the gradient is per-image, no eval/train split, no running stats. **First BN-backward in the attack path.** | no-BN: 7-layer product **L = 942K** (cert vacuous). BN: **cert N/A** — instance norm absorbs conv scale, Lipschitz is data-dependent (`γ·istd`) | **medium+** (done) | clean **70.0%** (no-BN) / **71.8%** (BN) → L∞ ε=0.1 → **0%**. BN trains *monotonically* (no-BN diverges late → best-θ rescue); BN is **more L2-fragile** (0.5% vs 8.8% @ L2 0.5). `runs/pgd_cifar{,_bn}_phase3.log` |
| **Imagenette** (r34/mnv2/enet/convnext/vit) | the **whole-net backward** → `dx`; reuse the proven `*_grad_floatBridges` machinery | product over ~50–100 layers ⇒ **astronomically vacuous** | **high (cost, not novelty)** | kernels 1–2 MB, iree-compile ~10–15 min each, PGD slow but runnable. Confirms fragility; cert meaningless |

**The attack is a "yes" all the way up** — it's the same proven-VJP-kernel recipe, just bigger. The
question is only whether the wall-clock is worth it past CIFAR (it mostly isn't — Imagenette PGD
tells you nothing the MLP didn't).

## 2. The certificate reality (why the naive ladder stops)

`Lip(f) ≤ ∏ᵢ ‖Wᵢ‖₂` (ReLU is 1-Lipschitz). Per-layer spectral norms sit around 2–4 after training,
so the product blows up geometrically: linear `L=5.3` (tight, exact), MLP `L=39` (vacuous),
a ~10-layer CNN `L~10³`, r34 `~10²⁰⁺`. Certified radius `= margin/(√2·L) → 0`. **Even *exact*
per-layer conv norms (FFT) don't help — the looseness is the PRODUCT, not the per-layer estimate.**
So:
- The naive-Lipschitz cert **demo is done** at linear-vs-MLP. CNN/CIFAR add a vacuous-with-conv
  data point (worth one rung to make the depth-cliff visual), but nothing past that.
- The thing to **formalize** is still the single, honest statement: `Lipschitz f L` + per-layer
  spectral-norm composition ⇒ a *sound* (if loose) certified radius — a real theorem, the
  verification payoff. Do it at the linear/MLP scale where it's checkable. ✅ **DONE** —
  `LeanMlir/Proofs/LipschitzCert.lean`.
- The product being loose isn't the end: you can **train for a small `L`**. ✅ **DONE 2026-06-28:
  `mnist-mlp-spectral`** — projected SGD onto `‖Wᵢ‖₂ ≤ c` (cap `L ≤ c³`) turns the MLP's vacuous
  cert non-vacuous (at c=2.0: cert@L2 0.5 `0% → 24%`, L∞ PGD `5.8% → 53%`, clean `97.8% → 96.2%`;
  c=1.5 peaks certified acc at 42%). A real accuracy↔robustness trade-off curve with an optimal `c`
  — the empirical face of the formalized radius. `runs/spectral_mlp_phase3.log`.

## 3. The real deep-net certificate: randomized smoothing (the Imagenette answer)

For nets past ~3 layers, the scalable certificate is **randomized smoothing** (Cohen, Rosenfeld,
Kolter 2019): define the smoothed classifier `ĝ(x) = argmax_c P[f(x+η)=c]`, `η ~ N(0,σ²I)`; then
`ĝ` is **certifiably robust** in L2 with radius `σ·Φ⁻¹(p_top)`, where `p_top` is the top class's
probability under noise. Properties that make it the Imagenette answer:
- **Architecture-agnostic, depth-independent** — no per-layer Lipschitz, no product. Works on r34/vit
  unchanged.
- **It's just forward passes** — a Monte-Carlo estimate of `p_top` over N noisy samples; reuses your
  existing forward kernels, no new backward, no new codegen. A *cheap* demo (no input-VJP kernel).
- **Honest cost**: the radius shrinks with `σ` vs accuracy (the noise–robustness trade-off), and the
  guarantee is high-probability (a confidence parameter), not deterministic. Still a real certificate.
- Alternative for trained-for-it nets: **IBP / interval bound propagation** (deterministic, but needs
  Lipschitz/IBP-aware training to be non-vacuous).

So the deep-net robustness demo isn't "bigger Lipschitz product" — it's a **randomized-smoothing**
driver (sample noise, forward, count, Clopper–Pearson lower bound on `p_top`, certified radius). That
would be a genuinely new, scalable result on the Imagenette verified nets, where the Lipschitz cert is
hopeless.

## 4. Recommendation / sequencing

1. ✅ **MNIST-CNN** (DONE 2026-06-28, `mnist-cnn-pgd`): exercised the conv/maxpool input-VJP kernel
   (the codegen milestone — `select_and_scatter`-back + transpose/reverse conv VJP, all run on the
   GPU through IREE), and the conv-aware cert made the depth-cliff visual (linear L=5.3 tight →
   MLP L=39 vacuous → CNN L=3196 more vacuous). Built on `cnn_train_step.mlir`'s backward ops.
2. ✅ **`Lipschitz f L` formalization** (DONE 2026-06-28, `LeanMlir/Proofs/LipschitzCert.lean`, 3-axiom
   clean): `lipschitz_margin_certified_radius` (Tsuzuku et al. 2018) — `L`-Lipschitz logit map +
   margin `m` ⇒ every `‖δ‖₂ < m/(√2·L)` keeps the argmax. The cert is now a *proof*, not a number.
   `LipschitzL2.comp` + `clm_lipschitzL2` prove the per-layer **product** `L = ∏‖Wᵢ‖₂` sound — and
   show why it's loose past one layer. The honest "this is verified DL."
2b. ✅ **Spectral-norm training** (DONE 2026-06-28, `mnist-mlp-spectral` + `mnist-cnn-spectral`): the
   gap-shrinking lever — projected SGD onto `‖Wᵢ‖₂ ≤ c` makes the formalized cert non-vacuous.
   **MLP**: cert@L2 0.5 `0% → 42%` (c=1.5), or `0% → 24%` at 96% clean (c=2.0). **CNN**: certifies
   only at a *smaller* radius — cert@L2 0.1 `0% → 54%` at 97% clean (c=2.0); reaching cert@L2 0.5
   needs collapsing to 29% clean. The CNN is harder (5-layer product `L ≤ c⁵` + the *loose* conv
   tap-sum over-penalizes convs when projected onto). Next research lever = tighter-than-product
   composition (the formalization makes it a concrete theorem-strengthening target); and the **exact
   (Sedghi–Gupta–Long FFT) conv spectral norm** — the spectral-CNN study showed the per-layer conv
   *estimate* (not just the product) is now the CNN's bottleneck.
3. **CIFAR**: optional; mostly BN-folding bookkeeping over the CNN.
4. **Imagenette**: **don't** chase the Lipschitz cert (vacuous). The attack is a cheap-ish add if
   you want the "deep verified nets are fragile" data point. The *certificate* there = a
   **randomized-smoothing demo** (forward-only, scales) — the real next robustness direction, and
   arguably more valuable than CNN/CIFAR Lipschitz.

## 5. References / in-repo anchors

- Madry et al. (PGD, 2017); Cohen, Rosenfeld, Kolter, *Certified Adversarial Robustness via
  Randomized Smoothing* (2019); Gowal et al. (IBP, 2018); Sedghi, Gupta, Long, *The Singular Values
  of Convolutional Layers* (2019, exact conv spectral norm); Tsuzuku et al. (Lipschitz-margin, 2018).
- In-repo: `LeanMlir/VerifiedTrain.lean` (`attackPgd`/`attackPgdMlp`/`genLinearPgdStep`/`genMlpPgdStep`/
  `specNormW` — the templates), `verified_mlir/cnn_train_step.mlir` (the CNN backward ops to mirror),
  the proven `convBack`/`maxPoolBack`/`*_grad_floatBridges` (the deep-net input-VJPs), `planning/robustness.md`.
