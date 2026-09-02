# Reuse audit: what in this repo duplicates Mathlib

Standing audit of the 537 committed `.lean` files in `targets.txt`, against the vendored
Mathlib at `.lake/packages/mathlib/Mathlib` (pin `v4.32.2`). 2026-09-02.

## Headline

**The Mathlib half came back clean.** Across 6,710 declarations in 537 files I found *no*
declaration that a Mathlib declaration replaces. Every finding below is category 5 —
near-clones inside this repo — and all but one sit **outside** `LeanMlir/Proofs/`.

Two pieces of evidence for the clean result, both cheap to re-run:

* **Name-level.** Intersecting the repo's 2,475 theorem/lemma names against Mathlib's
  128,628 yields 10 hits, all parse noise (`below`, `is`, `comp`, `mono`, …). The repo's
  naming is entirely domain-specific; nothing is a renamed Mathlib lemma.
* **Shape-level.** A 4-line-shingle near-clone pass over the 188 proof files finds exactly
  one pair above 35% overlap, and that pair is a documented deliberate analogue (below).

Where the proof tree touches standard mathematics it already cites Mathlib: `fderiv` and
`ContinuousLinearMap` (`Tensor.lean`), `Real.sum_le_exp_of_nonneg` / `Real.pi_gt_d20` /
`gaussianPDFReal` (`SmoothingPhiBounds.lean`), `Convex.norm_image_sub_le_of_norm_deriv_le`
and `Real.cosh_sq_sub_sinh_sq` (`GeluLipschitz.lean`), `tendsto_atTop_ciSup` and
`isFixedPt_of_tendsto_iterate` (`MuonNewtonSchulz.lean`), `finProdFinEquiv` (`Tensor.lean`).

---

## Findings

### 1. `tests/TestMHSA.lean:146-211` and `tests/TestSDPA.lean:110-186` — the gradcheck harness, re-copied

**Replace with:** `ViTGradcheck.{pow10, digitsToNat, splitAtChar, parseFloat, parseResults, runFn, randVec, dot, axpy}` (`LeanMlir/GradcheckHelpers.lean`)
**Why it matches:** verified byte-identical after stripping `private` — the only diffs across all
eight functions are one trailing comment (`-- y + a·x` on `axpy`), one alignment space in
`parseFloat`, and one comment line in `parseResults`. `GradcheckHelpers.lean`'s own module
docstring names these two files as its consumers ("Used by the ch10 ViT de-risk tests
(TestSDPA/TestMHSA/TestViTBlock)"), and the two siblings `TestViTBlock.lean` and
`TestViTTiny.lean` already `import LeanMlir.GradcheckHelpers` and keep only a local
`compileCheck`. These two never migrated.
**How:** add `import LeanMlir.GradcheckHelpers` + `open ViTGradcheck`, delete the local block.
`GradcheckHelpers.lean` has zero imports, so this costs nothing in build surface.
**Caveat:** `TestSDPA`'s `runFn` is a genuine specialization (takes `Array (Array Float)` and
formats a fixed `{Nn}x{Dd}xf32` shape via `fmtInput`) — keep it local; the other seven go.
**Deletes:** 66 lines from TestMHSA, ~62 from TestSDPA.
**Confidence:** verified

### 2. `Bestiary/*.lean` (41 files) — `private def summarize` in every entry

**Replace with:** one `summarize` in `LeanMlir/Spec.lean` (which all 41 already import)
**Why it matches:** all 41 Bestiary entries define an 11-line `private def summarize (spec : NetSpec) : IO Unit`.
There are exactly four bodies, differing only cosmetically: params printed in M vs K units vs
bare; `input : {h} × {w}` vs `context : {h} tokens` for the text nets; and the `validate : OK`
message. No shared `summarize` exists anywhere in `LeanMlir/`.
**How:** `def summarize (spec : NetSpec) (unit : ParamUnit := .M) (inputLabel : String := "input") : IO Unit`
in `LeanMlir/Spec.lean`; each entry drops its copy.
**Deletes:** ~451 lines.
**Confidence:** verified

### 3. `tests/*.lean` (33 files) — `private def mkParam`

**Replace with:** one `mkParam` in `LeanMlir/VerifiedNets.lean` (imported by all 33)
**Why it matches:** 24 of the 33 share a byte-identical 8-line body (two md5 groups that differ
only by trailing `-- BN γ` / `-- BN β / biases` comments). Four more (`TestR50AccumTie`,
`TestR50AccumShardTie`, `TestR50BceTie`, `TestR50LambTie`) use a genuinely different fan-in
convention (fan-out for rank-4, fan-avg for rank-2) — that is a second function, not noise.
The remaining five are one-off variants.
**How:** `mkParam` + `mkParamFanAvg` in `LeanMlir/VerifiedNets.lean`; the 28 files delete theirs.
**Deletes:** ~192 lines directly, ~264 if the variants are folded in.
**Confidence:** verified

### 4. `tests/TestCifar8WideTrain.lean` — a pre-parameterization copy of `TestCifar8AdamTrain.lean`

**Replace with:** the width parameter `TestCifar8AdamTrain.lean` already has
**Why it matches:** 573 of the Wide file's 659 lines are byte-identical to the Adam file. The
real differences are three: `D1 := 512` vs `64`, the emitted func names slugged `cifar8w_*`,
and `emitAdamV` vs `emitAdamVDP` (the Wide file predates the replica plumbing). Decisively,
`TestCifar8AdamTrain.lean:325` already carries `cifar8BnAdamBody (d1 : Nat := D1)` with the
comment *"shadow the module default so the dense head width is sweepable"*, and
`cifar8BnAdamTrainStep (d1) (fname)` already takes the func name as a parameter — the exact
mechanism that makes the Wide file redundant exists in the file it was copied from.
**How:** instantiate the Adam file's parameterized renderers at `d1 := 512`, `fname := "cifar8w_…"`;
delete the Wide file's duplicated body (`convFwd`, `convBack`, `convWGrad`, `scatter`,
`maxpoolFwd`, `bnFwd`, `bnBack`, `bnParamGrad`, `params`, `adamConsts`, `selMask2/4`,
`emitSgd`, `emitMomentum`, `tryCompile`, `cifar8AdamBody`).
**Deletes:** ~570 lines.
**Confidence:** verified

### 5. `LeanMlir/Proofs/Architectures/MnistCNN.lean:628-767` — `Spatial` re-proves `Mini`

**Replace with:** a `Mini`/`Spatial`-shared core parameterized on the conv closed forms
**Why it matches:** the only proof-tree finding. `Mini` (458–613, 1×1 kernels) and `Spatial`
(628–767, 3×3 center-tap kernels) are two instances of the same 2-channel / 8-pool-window /
10-class CNN. Seven theorems are byte-identical across the two namespaces —
`poolTensor_inj` (519/689), `blockZ_eq` (551/715), `block1_eq` (543/709), `conv1_pos`
(491/663), `pooled_pos` (565/727), `dense3_pos` (572/733), `pooled_eq` (559/722) — plus
`flatConv1_eq`, `convZ_eq` and the two capstones, which differ only in the definition name.
Everything downstream of `conv1_eq`/`conv1_pos`/`conv2_eq`/`conv2_pos` depends on `W1`/`W2`
*only through those four lemmas*; the genuine difference is confined to the kernel defs, the
three `hW1`/`hW2`/`W1_center` helpers, and the `conv2d_1x1` vs `conv2d_center3x3` rewrites.
**How:** a `section` taking `conv1_eq`/`conv1_pos`/`conv2_eq`/`conv2_pos` as `variable`
hypotheses, instantiated twice — the same shape the file already uses for
`mnistCnnNoBn_has_vjp_at`'s hypothesis discharge.
**Deletes:** ~85 of `Spatial`'s 140 lines.
**Confidence:** verified

### 6. `tests/*.lean` — the remaining per-test scaffolding

**Replace with:** shared definitions in `LeanMlir/VerifiedNets.lean` / `LeanMlir/GradcheckHelpers.lean`
**Why it matches:** same pattern as findings 1 and 3, at smaller multiplicity. Counts of files
carrying a private copy: `compileCheck` 13, `tryCompile` 8, `convWGrad` 6, `mkLabels` 5,
`sgd` 5, `smoke` 5, `biasGrad` 4, `entryAndBatch` 2, `upsampleFlat` 2. The clone pass
confirms exact-body groups within each (e.g. `compileCheck` splits 8 / 3 / 2).
**How:** lift per group; the bodies within each exact group need no parameterization.
**Deletes:** ~250 lines.
**Confidence:** verified (duplication); the grouping into shared signatures is a design call.

### 7. `demos/`, `apps/`, `jax/` — model definitions duplicated across driver mains

**Replace with:** one definition per architecture in a shared module
**Why it matches:** the same `NetSpec` literal appears in several driver mains: `resnet34` in 9
files, `tinyCifarDdpm` in 6, `unetPets` and `runIree` in 4 each, `autoencoderPets`,
`tinyDdpmUnet` and `loadCifar` in 3 each, and `r34UnetBratsOf`, `vgg16bn`, `mobilenetV3Large`,
`efficientNetV2S` in 2 each. Several pairs straddle `jax/MainX.lean` and
`apps/baselines/MainXTrain.lean` with byte-identical bodies.
**How:** a `demos/Models.lean` (or extend `Bestiary`) holding the specs; mains import it.
**Deletes:** ~400 lines.
**Confidence:** verified (duplication). Lower priority than 1–5: these are one-screen `NetSpec`
literals, and a shared module couples the jax/ and apps/ driver trees that are currently
independent. Worth doing for the `resnet34`×9 and `tinyCifarDdpm`×6 clusters at least.

---

## Checked and deliberately **not** flagged

* **`Proofs.Mat.{mulVec, outer, mul, transpose}`** (`Foundation/Tensor.lean:53-64`) vs
  `Matrix.{mulVec, vecMulVec, HMul, transpose}`. `Mat m n := Fin m → Fin n → ℝ` *is*
  `Matrix (Fin m) (Fin n) ℝ` definitionally (`Matrix` is `m → n → α`,
  `LinearAlgebra/Matrix/Defs.lean:55`) and all four ops match up to `rfl`. But
  `LeanMlir/Proofs/Codegen/MatBridge.lean` already proves exactly this correspondence
  (`Mat.mulVec_eq`, `Mat.transpose_eq`, `Mat.outer_eq`, `Mat.mul_eq`) and its header states
  the opt-in rationale: keeping `Tensor.lean`'s import surface free of `Matrix`. Decision
  already made and documented.
* **`Proofs.LipschitzL2`** (`Certificates/LipschitzCert.lean:41`) vs `LipschitzWith`.
  Mathlib's is `ℝ≥0∞`-valued with an `ℝ≥0` constant; the repo's `L : ℝ` comes from numeric
  `specNormW` output. `LipschitzWith.of_dist_le_mul` / `dist_le_mul` do bridge it, but the
  coercion friction on a computed `L` is real and the file says so at the definition.
* **`LeanMlir/Proofs/Foundation/UpstreamDraft.lean`** vs `planning/mathlib_upstream_drafts/PR{1,2}*.lean`.
  Byte-identical by design (the only diff is a needed `open Set`), and the mechanism works:
  the drafts are a `Certs` root (`lakefile.lean:780`) audited by `tests/AuditAxioms.lean:178`.
  Confirmed the pinned Mathlib has **none** of `strictMono_cdf`, `cdf_pos`, `cdf_lt_one`,
  `cdf_mem_Ioo`, `continuous_cdf`, `leftLim_cdf` (`Mathlib/Probability/CDF.lean` stops at
  `cdf_le_one` / `monotone_cdf`), nor any `cdf_gaussianReal_*`. Genuinely new material.
* **`Architectures/DepthwiseBackCertifiedTie.lean`** vs `Foundation/IR.lean` — the one
  shape-level near-clone pair (63%). The shared text is `Finset.sum_bij'` index arithmetic;
  the summand genuinely differs (depthwise drops the `Σ co` cross-channel sum), and the
  docstring names it "the depthwise twin of `IR.convBackDenote_eq_input_grad_formula`".
* **`tests/comparator/{Challenge,Solution}.lean`** — the duplication *is* the artifact.
  `leanprover/comparator` needs the statement twice (once `sorry`'d over Mathlib alone, once
  discharged); the header explains it.
* **`mnist-lean4/` (4 files), `historical/`** — a preserved lineage, not live code. README:82
  calls `mnist-lean4/` "Phase 1 — Pure Lean 4". The four snapshots progressively extend each
  other (`Main_working_1d_s4tf` → `_2d_` → `_cifar_s4tf_working_v1` → `_cifar_v2`); freezing
  them is the point.
* **Muon** (`Foundation/MuonGeometry.lean`, `MuonNewtonSchulz.lean`) — checked Mathlib for a
  von Neumann trace inequality, an SVD, a matrix polar decomposition, a Frobenius `Inner`
  instance on `Matrix`, and a "nearest orthogonal matrix" result. **None exist.** The only
  matrix inner products are `PosSemidef.toMatrixInnerProductSpace` (M-weighted,
  `Analysis/Matrix/Order.lean:336`). Genuinely new material.
* **`GeluLipschitz.lean`'s `one_sub_tanh_sq_le`** — Mathlib has `Real.tanh_sq_lt_one`
  (`Analysis/Complex/Trigonometric.lean:864`) but no `1 − tanh² = 1/cosh²` and no
  `exp|y|/2 ≤ cosh y`. The one micro-substitution available is
  `h1t2 : 0 ≤ 1 - t^2` ← `sub_nonneg.mpr (Real.tanh_sq_lt_one u).le`, replacing an `nlinarith`.
  Not worth a finding on its own.

## What has been applied

Landed on `mathlib-reuse-audit` (98 files, **−1327/+346**, net **−981 lines**); every touched target
rebuilt and the Bestiary golden guard re-run (189 variants, PASS).

| finding | status | shared home |
|---|---|---|
| 1 · gradcheck harness | ✅ done (−124) | `LeanMlir/GradcheckHelpers.lean` (already existed) |
| 2 · Bestiary `summarize` | ✅ done (41 files) | `NetSpec.summarize` in `LeanMlir/Spec.lean` |
| 3 · `mkParam` | ✅ done (33 files, −314) | `LeanMlir/VerifiedTrain.lean` |
| 6 · `compileCheck`/`tryCompile` | ✅ done (21 files, −195) | `LeanMlir/Types.lean` |
| 4 · Cifar8Wide | ⚠ superseded — see below | |
| 5 · MnistCNN `Mini`/`Spatial` | ⏳ not mechanical — see below | |
| 6 · rest (`mkLabels`, TrainPC kit) | ⏳ needs file-local constants lifted too | |
| 7 · model defs | ⏳ not started | |

Behaviour was preserved everywhere except three deliberate normalizations, none observable to a
gate: Evoformer's summary column width (13 → 12, matching all 40 siblings), QANet's and VAE's
`(~N K)` → `(~NK)`, and `iree-compile`'s stderr echo width (2000/5000 → 3000) — a truncation
length on a diagnostic string.

### `mkParam`: the divergence is now named rather than erased

`planning/lean_434_and_cleanup.md` §4.1 already had this one, with a fact the audit's own pass had
missed: the copies do not merely duplicate the canonical `VerifiedTrain.mkParam`, they **disagree**
with it. Measured across the 33: 25 use He **fan-IN**, the canonical uses He fan-OUT + Glorot, and
`tests/TestSgdRenderTie.lean` documented itself as "identical to the driver's `VerifiedTrain.mkParam`"
while differing from it.

Unifying the formula would change the initial θ of 25 gates, so it is **not** what landed. Instead:
`mkParam` is un-`private`d and serves the 4 R50 gates that already matched it byte-for-byte;
`mkParamHeFanIn` holds the 25-gate consensus with the disagreement stated in its docstring;
`biasSigma` covers the two ties that need non-zero biases; and `TestR34DpShard` /
`TestR50GradCheck` keep local copies, now named `mkParamFanOutFlat` / `mkParamBiasSeeded` for
what they actually are. The false docstring is gone with the copy it sat on.

**Open decision for a human:** should those 25 gates move to the driver's init? That is a
behaviour change to 25 gates, not a refactor, and it wants a deliberate re-run rather than a
dedup commit.

### Finding 4 is superseded by something worse

`tests/TestCifar8AdamTrain.lean` and `tests/TestCifar8WideTrain.lean` are **not build targets** —
neither is a `lean_exe` root, neither has ever produced an `.olean`, and nothing imports them.
They still compile under `lake env lean`, and every render in both is marked
*"RETIRED 2026-07-30 (§2i) … NOT written"*, superseded by `Proofs/Codegen/CnnRender.lean`. Their
stated remaining purpose is "the tie reference", but no gate mechanically consumes them.

So the 573 duplicated lines are the smaller problem: **1,359 lines of unbuilt, unreferenced Lean**
is the finding. Deduplicating them would also need a shared module that is itself a build target
(an unbuilt module cannot be imported), i.e. a lakefile change in service of dead reference
material. Retiring or properly wiring them is a call for a human, not a dedup commit.

### Finding 5 is proof surgery, not a lift

The seven byte-identical `Mini`/`Spatial` theorems are identical as *text* but elaborate against
different definitions, and several unfold the concrete kernels — `poolTensor_inj` finishes with
`fin_cases ci <;> (simp [W2, b1, b2] at heq; linarith)`, which needs `W2`'s literal entries. A
shared section therefore cannot just take `conv1_pos`/`conv2_pos` as hypotheses; it needs the
affine form `conv2 ci r s = A ci + B ci · T0 0 r s` with `0 < B ci` hypothesized, and each
namespace discharging it. That is a real (worthwhile) ~130-line refactor, and the only one of the
seven findings that touches a proof — so it wants its own commit and its own `lake build Certs`.

## Summary

Examined 6,710 declarations across 537 files; 188 of them proof files. **Zero are replaceable
by a Mathlib declaration** — the proof tree already cites Mathlib wherever it touches standard
mathematics, and the two places where it defines its own version of a Mathlib notion
(`Mat`'s operations, `LipschitzL2`) are documented decisions with the bridge lemmas already
written. All seven findings are in-repo duplication, six of them in `tests/`, `demos/` and
`Bestiary/` rather than in the proofs: roughly **2,050 deletable lines**, of which ~1,150 sit
in findings 1–5 and are mechanical (one shared module already exists and is already imported
by the files that would use it). Findings 4 and 5 are the interesting ones — in both cases the
generalization that makes the copy redundant is *already written in the file that was copied*.
