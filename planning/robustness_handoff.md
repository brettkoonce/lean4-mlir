# Robustness thread — session handoff (2026-06-28)

A pick-up doc for the next session. This session took the robustness ladder
(`planning/robustness.md`, `planning/robustness_ladder.md`) from linear/MLP up through CIFAR +
BatchNorm, formalized the certificate as a theorem, and added the gap-shrinking training lever.
**Everything below is committed and builds green** (gfx1100 / IREE rocm).

> **UPDATE 2026-06-28 (follow-on session): §6.1 randomized smoothing is now BUILT** —
> `mnist-mlp-smooth` / `mnist-cnn-smooth` / `cifar-smooth` + `VerifiedNet.smoothCertify`
> (forward-only, depth-independent). Cohen 2019: noise-augment training (host-side `N(0,σ²I)`,
> `lean_f32_add_gaussian_tiled`, graph untouched) → sample `n=1024` → Clopper–Pearson lower-bound
> `p_A` (hand-rolled probit + incomplete-beta, **validated to 6 dp vs scipy**) → radius `σ·Φ⁻¹(p_A)`.
> It certifies a *non-vacuous* L2 radius on every rung where the Lipschitz product was vacuous
> (tightened to Cohen's large-`n=10112` regime — n is the honest lever, ceiling `σ·Φ⁻¹(α^(1/n))` rose
> `2.47σ→3.20σ` vs the first `n=1024` run; cert@1.5 newly non-zero; ACR up ~25–30%). At σ=0.5,
> cert@0.5/1.0/1.5: CIFAR-CNN (spectral 0% @ L=942K) → **32.5/18.5/8.0%** (ACR 0.40); MNIST-CNN →
> **93/78/48.5%** (ACR 1.28); MNIST-MLP → **87/67.5/31%** (ACR 1.13). Run across both gfx1100 GPUs
> (`run_smooth_2gpu.sh`, HIP_VISIBLE_DEVICES per stream, ~35 min).
> See `planning/robustness_ladder.md` §4.3b for the full table; logs `runs/smooth_{mlp,cnn,cifar}.log`.
> (Driver committed f30f857.) The cert paradigm is now complete on both ladders.
>
> **And the smoothing radius is now a THEOREM** (`LeanMlir/Proofs/LipschitzCert.lean`, 3-axiom clean,
> alongside `lipschitz_margin_certified_radius`): `smoothing_certified_radius` — the `σ·Φ⁻¹(p_A)`
> radius proved as the same Lipschitz-margin argument on the per-class probit score fields
> `Φ⁻¹∘P[f(x+η)=·]` (each `(1/σ)`-Lipschitz — the Cohen/Salman Gaussian-isoperimetry content, taken as
> a hypothesis exactly as `L` is in the Tsuzuku theorem; `smoothed_margin_certified_radius` = the core
> `σ·m/2` step giving Cohen's `(σ/2)(Φ⁻¹(p_A)−Φ⁻¹(p_B))`). Both certificates of `cert ≤ TRUE ≤ PGD`
> are now proofs. (UNCOMMITTED — awaiting sign-off.)

## 0. The one-paragraph state of the world

The **attack ladder** (PGD via the proven input-VJP, run as an IREE kernel) now reaches
**linear → MLP → MNIST-CNN → CIFAR-CNN → CIFAR-CNN+BN**, the last via the first **BN-backward** in
the attack path. The **certificate** went from a number → a **theorem** (`LipschitzCert.lean`,
3-axiom clean) → a thing you can **train for** (spectral-norm projected SGD) → and we mapped exactly
where it breaks (the conv tap-sum product is unfixable past the MLP). The attack always wins (every
net collapses to ~0% at small ε); the cert is tight at 1 layer and vacuous beyond, which is the whole
Thread-4 thesis shown end-to-end on the verified nets.

## 1. What got built this session (commits, newest→oldest)

| commit | what | headline |
|---|---|---|
| `929ebed` | **CIFAR-BN PGD** (`cifar-bn-pgd`, `genCifarBnPgdStep`) | first BN-backward in the attack path (instance-norm grad-input 3-term formula ×4); clean 71.8% → 0% @ L∞ 0.1; **cert N/A** |
| `5cba72c` | **CIFAR PGD** (`cifar-pgd`, `genCifarPgdStep`) + generic-driver refactor + **best-θ** | 70.0% → 0%; global L = **942K** (7-layer product); depth-cliff 5.3→39→3196→942K |
| `0169573` | **CNN spectral** (`mnist-cnn-spectral`) | cert 0%→53.7% @ L2 0.1 (c=2.0, −2% clean); CNN certifies only at smaller radius |
| `7bb1bfb` | **MLP spectral** (`mnist-mlp-spectral`) | cert 0%→24% @ L2 0.5 + L∞ PGD 5.8%→53% for −1.6% clean; the gap-shrinking lever |
| `c465053` | **Lipschitz-margin theorem** (`LeanMlir/Proofs/LipschitzCert.lean`) | `lipschitz_margin_certified_radius` (Tsuzuku 2018), 3-axiom clean — the cert is now a proof |
| `369b437` | **MNIST-CNN PGD** (`mnist-cnn-pgd`, `genCnnPgdStep`) | conv/maxpool input-VJP via IREE; 99% → 0% @ L∞ 0.3 |

(Prior, pre-session: `1971ce7` MLP PGD, `d01f7e8`/`d883657` linear PGD+cert.)

## 2. Code map — where everything lives

All the host/codegen code is in **`LeanMlir/VerifiedTrain.lean`**:
- **PGD-step kernel generators** (`gen<Net>PgdStep : Nat → Float → Float → Bool → String`, emitting
  StableHLO text — `bs`, `eps`, `alpha`, `linf`): `genLinearPgdStep`, `genMlpPgdStep`,
  `genCnnPgdStep`, `genCifarPgdStep`, `genCifarBnPgdStep`. Each = forward (saving the pre-acts the
  backward needs) → softmax-CE seed `g = softmax−onehot` → full input-VJP to `dx` → L∞ sign-step / L2
  projected-step + ε-ball project + [0,1] clip. **They mirror the committed `<slug>_train_step.mlir`
  backward**, then add the final conv1 input-VJP the train step omits (it only needs weight grads).
- **Generic drivers** (the refactor): `VerifiedNet.attackPgdConvNet (… genKernel) (withCert := true)`
  = train (best-θ) → clean → L∞ sweep → cert (conv-aware product, skippable) → L2 sandwich.
  `attackPgdSpectralConvNet (… caps genKernel)` = the projected-SGD cap sweep. Thin wrappers:
  `attackPgdCnn/Cifar/CifarBn`, `attackPgdSpectralCnn/Cifar`. (linear/mlp keep their own
  `attackPgd`/`attackPgdMlp`/`attackPgdSpectralMlp`.)
- **Cert helpers**: `specNormW` (power iteration on the `WᵀW` Gram — high-precision, cert path),
  `specNormMV` (matrix-free power iteration — cheap, training-time projection), `specNormGet`
  (strided, for conv taps), `specNormConvTapSum` (sound conv operator-norm bound
  `‖T‖₂ ≤ Σ_tap‖W[:,:,ky,kx]‖₂`), `projectSpectral` (rescale dense `‖W‖₂` + conv tap-sum to ≤ c).
- **Best-θ**: `attackPgdConvNet` evals each epoch and attacks the highest-accuracy checkpoint
  (rescues plain-SGD late divergence on CIFAR; the per-epoch trajectory is printed).

The **theorem** is `LeanMlir/Proofs/LipschitzCert.lean` (namespace `Proofs`, imported by
`LeanMlir.lean`, audited in `tests/AuditAxioms.lean`):
- `lipschitz_margin_certified_radius` — `L`-Lipschitz logit map + margin `m` ⇒ every `‖δ‖₂ < m/(√2·L)`
  keeps the argmax. Over `EuclideanSpace ℝ (Fin k)` (genuine L2 + the √2).
- engine: `logit_gap_stable` (a pairwise logit gap is `√2·L`-Lipschitz), `coord_pair_bound`
  (`(vᵢ−vⱼ)² ≤ 2‖v‖²`), `euclid_norm_sq`. composition: `LipschitzL2.comp` (constants multiply),
  `clm_lipschitzL2` (linear layer is `‖A‖`-Lipschitz). All 3-axiom clean.

**Mains + exes** (`apps/{mnist,cifar}/Main*.lean`, `lakefile.lean`): `mnist-{linear,mlp,cnn}-pgd`,
`mnist-{mlp,cnn}-spectral`, `cifar-pgd`, `cifar-spectral`, `cifar-bn-pgd`. Env overrides for cheap
smokes: `CIFAR_PGD_EPOCHS`, `SPECTRAL_EPOCHS`, `CNN_PGD_EPOCHS`.

**Run recipe** (all): `PATH=$PWD/.venv/bin:$PATH IREE_BACKEND=rocm .lake/build/bin/<exe> data`.
Canonical logs: `runs/pgd_{cnn,cifar,cifar_bn}_phase3.log`, `runs/spectral_{mlp,cnn}_phase3.log`.

## 3. The results, in one table

| net | clean | L∞ ε=0.1 | L∞ ε=0.3 | L2 ε=0.5 | global L | cert |
|---|---|---|---|---|---|---|
| linear | 92.1% | 24.0% | 0% | 79.4% (PGD) | **5.29** (exact) | **53.3%** @ L2 0.5 |
| mlp | 97.8% | 5.8% | 0% | 86.0% (PGD) | **39.2** | 0% (vacuous) |
| mnist-cnn | 99.0% | 57.8% | **0%** | 95.8% (PGD) | **3,196** | 0% |
| cifar-cnn | 70.0% | **0%** | 0% | 8.8% (PGD) | **942,040** | 0% |
| cifar-cnn-bn | 71.8% | **0%** | 0% | 0.5% (PGD) | n/a | **N/A** (data-dependent) |

**Spectral training** (the lever): MLP c=2.0 → cert 0%→24% @ L2 0.5 for −1.6% clean (c=1.5 peaks 42%).
CNN c=2.0 → cert 0%→53.7% @ L2 **0.1** for −2% clean (certifies at a tighter radius). CIFAR: **no cap
works** — c=6/4/3 train (45–55%) but L stays ≥1357; c≤2 kills the 7-layer net.

## 4. Findings worth carrying forward

1. **The depth-cliff is real and now quantified**: global L = 5.3 → 39 → 3,196 → 942K. Even *exact*
   per-layer conv norms wouldn't save it — the looseness is the **product**, not the per-tap estimate.
2. **You can train for a small L** (spectral projection), and it genuinely shrinks the cert↔PGD gap on
   the MLP/CNN — but the deeper the net, the tighter the cap must be and the more clean accuracy it
   costs, until (CIFAR, 7 layers) no cap is simultaneously trainable and small-L.
3. **`cifar_bn`'s "BN" is instance normalization** (per-image over spatial, `nf=H·W`) — so the attack
   gradient is per-image, there's no eval/train split, and `cifar_bn_fwd` is the deployed forward.
   This made the BN attack tractable.
4. **BN stabilizes training but increases L2 fragility**: it trained monotonically to 71.8% (the no-BN
   net diverged at epoch 10, rescued by best-θ) yet is *more* L2-fragile (0.5% vs 8.8% @ L2 0.5) —
   instance norm's `istd` amplifies low-variance perturbations.
5. **The BN cert is a separate problem**: instance norm absorbs the conv weight scale (scaling a conv
   before BN does nothing), and its Lipschitz is data-dependent (`γ·istd`, unbounded as variance→0).
   So `attackPgdCifarBn` runs with `withCert := false`.

## 5. Gotchas learned this session (save the next session time)

- **`s!`-string brace escaping**: in Lean interpolated strings, `}` is literal-by-default and `{`
  escapes as `\{`. So `}}` produces **two** literal braces (a bug), and `{{` tries to interpolate.
  The MLIR region syntax `({ … }) {attr}` must be written `(\{ … }) {attr}` (the closing `})` is bare,
  the attr-dict `{` is `\{`, or interpolate a whole `let poolAttr := "{…}"` variable). Hit this twice.
- **Stale binary**: after editing a kernel generator, a combined `lake build … && run` in one Bash
  call can run the *old* binary if the build is async/cached oddly — rebuild explicitly and check the
  binary mtime vs source before trusting a failure.
- **Plain SGD diverges on the deeper nets**: CIFAR-CNN peaked 70% @ epoch 8 then went NaN (cert = NaN,
  PGD = random 10%). **best-θ checkpointing** (now in `attackPgdConvNet`) fixes it; BN avoids the
  divergence entirely.
- **CIFAR cert eval is slow**: `specNormW` on the 4096×512 dense is ~1e9 host ops (~minutes). Fine for
  one cert call, painful in a cap sweep.
- **CIFAR eps**: the driver sweeps the MNIST eps `[0.1, 0.2, 0.3]`; for CIFAR these are ≥3× the
  standard 8/255≈0.031 benchmark, so it's all 0%. If you want a CIFAR degradation *curve*, thread a
  per-net eps list (small change to `attackPgdConvNet` — make the L∞/L2 eps lists parameters).

## 6. Open next steps (pick one)

Ranked by interest / novelty, with concrete entry points:

1. **Randomized-smoothing certificate** (`planning/robustness_ladder.md` §3 — the *real* deep-net
   cert, and the answer where the Lipschitz product is hopeless). Cohen–Rosenfeld–Kolter 2019:
   `ĝ(x)=argmax_c P[f(x+η)=c]`, `η~N(0,σ²I)` ⇒ certified L2 radius `σ·Φ⁻¹(p_top)`. **Forward-only**
   (no new kernel, no input-VJP) — sample N noisy copies, run the existing `<slug>_fwd`, count,
   Clopper–Pearson lower-bound `p_top`, report radius. Architecture-agnostic + depth-independent, so it
   works on *every* net including CIFAR/Imagenette where the product is vacuous. Cheapest high-value
   demo; could be `<slug>-smooth` reusing `forwardF32`. **Recommended next.**
2. **Formalize the conv/BN composition** (extend `LipschitzCert.lean`): the certified-radius theorem
   is generic; add the per-layer pieces — conv-as-linear-map `‖·‖`, ReLU/maxpool 1-Lipschitz (in L2),
   and compose to a *whole-net* `LipschitzL2 (∏‖Wᵢ‖) f` theorem. Makes "the product cert is sound"
   a proof for the actual CNN, not just abstractly. Pure Lean/Mathlib, 3-axiom target.
3. **Tighter-than-product composition** (the real research lever): the formalization now makes
   "tighten the bound" a concrete theorem-strengthening target (e.g. Fazlyab semidefinite bounds,
   or LipSDP). Bigger lift; the payoff is a non-vacuous deep-net Lipschitz cert.
4. **Exact (Sedghi–Gupta–Long FFT) conv spectral norm**: the spectral-CIFAR study showed the per-layer
   conv *estimate* (tap-sum, loose by √9/conv) is the CIFAR bottleneck, not just the product. Replace
   `specNormConvTapSum` with the exact SGL value (2-D FFT of the kernel, max singular value over
   frequencies) — lets spectral projection constrain to the same L with √9× larger (truer) conv
   weights, so the constrained CIFAR net might actually train + certify.
5. **Imagenette attack** (`r34/mnv2/enet/convnext/vit`): mechanically possible (thread the proven
   whole-net backward to `dx`), but expensive (1–2 MB kernels, ~10–15 min iree-compile each) and only
   reconfirms fragility. Low novelty per the ladder doc — do only if you want the data point.
6. **Per-net eps + CIFAR curve**: small polish — parameterize the L∞/L2 eps lists in
   `attackPgdConvNet` so CIFAR shows a real degradation curve at 8/255-scale ε instead of all-0%.

## 7. References / anchors

- `planning/robustness.md` (Thread-4 status log, every rung), `planning/robustness_ladder.md` (the
  two-ladders framing + the rung table), this doc (session handoff).
- Madry 2017 (PGD); Tsuzuku–Sato–Sugiyama 2018 (Lipschitz-margin radius — the formalized theorem);
  Cohen–Rosenfeld–Kolter 2019 (randomized smoothing); Sedghi–Gupta–Long 2019 (exact conv norm).
- In-repo: `LeanMlir/VerifiedTrain.lean` (all kernels + drivers), `LeanMlir/Proofs/LipschitzCert.lean`
  (the theorem), `verified_mlir/{cnn,cifar,cifar_bn}_train_step.mlir` (the backward to mirror),
  `tests/AuditAxioms.lean` (3-axiom audit).
