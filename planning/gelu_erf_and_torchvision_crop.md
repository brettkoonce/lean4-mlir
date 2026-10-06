# Removing items 3 and 4 from §9.6's ledger: the exact-erf GELU and torchvision's crop sampler

Written 2026-10-06, after the post-C6 ViT-Ti pair landed (`runs/2026-10-01-vit-{jax,verified}-bf16-300ep/`,
§9.6). The chapter's "what does not match" list is down to six items; two of them are code we
could write rather than hardware or a decision:

* **Item 3.** The GELU is the tanh approximation on both paths (Definition `ax:geluScalar`); DeiT's
  and ConvNeXt's is the exact `x · Φ(x)`. `vit_parity_todo.md` P-A, `imagenet_parity.md` D1.
* **Item 4.** RandomResizedCrop is TensorFlow's `sample_distorted_bounding_box`, not torchvision's
  sampler. `vit_parity_todo.md` P-G, `imagenet_parity.md` F2.

Both are "both paths" items: the verified path inherits them from the shared recipe and pipeline,
so closing either moves the reference and the verified path together and changes nothing about any
pair's agreement claim. Both also appear outside chapter 9 — ch 8's ConvNeXt ledger
(`content.tex` ~11002–11007), the still-out lists of §8.6 and §9.7, ch 5's A2/A1 ledger (the crop,
~7098), appendix A (~17779) — and the same two fixes close every one of those lines.

**Where things are (2026-10-06, end of session).** §5's decisions are made. Item 4 (§2) is done in
code and gated, committed on `wp8fg` (aa974a95), not pushed; no run carries it. Item 3 (§3): the
probe and the proofs (0ce30dc7) and the op, the nets and the ties (10fc13ee) are committed, not
pushed; §3.7's JAX flag is in, set in the six ImageNet ViT and ConvNeXt configs and gated. The next
step is §3.8, the committed exact-GELU renders: until it lands the ImageNet references compute the
exact GELU and the committed ImageNet renders the tanh one. No runs until both items are in the code
(§5 b), then the ViT-Ti and ConvNeXt-T reruns go first, ahead of the owed R34 / R50 reruns and the
side-quest queue.

## 1. Order

**Crop first, GELU second.** The crop is a day of data-side work with no proof, and it has a
deadline: the R34 / R50-2018 / R50-A3 reruns the book owes (`r34-default-jax-4gpu`,
`r50-2018-jax-4gpu`, `r50-a3-jax-4gpu` and their verified rows, de7f7857) and the side-quest
queue (A2/A1, MNv4 `full`, ConvNeXt-S/B, ViT-S/B; `side_quest_runs.md` §1, ~78 days) have not
launched. Every one of those that starts before the crop lands needs a third rerun to close
item 4; every one that starts after carries it for free. The GELU is multi-day and touches the
proofs; its reruns are ViT and ConvNeXt only, and the S/B jobs in that queue carry it too if it
lands before they start.

## 2. Item 4: torchvision's RandomResizedCrop

**Status 2026-10-06: in the code, no run yet.** `TrainConfig.cropTorchvision`, `trainResize`
(`.bicubic` / `.bilinear` / `.random`) and `cropFallbackCenter` (§5 c, d). On: R34 and R50-2018
bilinear; every RSB tier and MNv4 timm's `random`; ConvNeXt T/S/B and ViT Ti/S/B bicubic. B0 takes
EfficientNet's centre-crop fallback; MNv2 is byte-identical. 70 of the 74 `jax/generated/` files
move. `scripts/gates/crop_sampler_gate.py`: 200k draws a shape against torchvision 0.28's
`get_params` at 375², 500×375, 360×640 and 500×200, KS ≤ 0.004 on scale and log aspect, fallback
rate equal (0.0122 at 2:5); TF's sampler and a uniform-aspect sampler both red; B0's fallback gives
the centre window on 298 of 300 draws at 120×1600 and never the whole image; TF's antialiased
bilinear is within 0.13–0.20 mean / 1 max grey level of PIL's. `pc_crop` in
`scripts/lib/precheck.sh` is in all 39 ImageNet job confs (all pass DRY_RUN; a pre-change shim
is refused). `mixup_gate.py`'s gate-1 pins re-pinned (they were already stale before the change);
`bce_target_gate.py` green. `scripts/parity/identity.py`, named in §2.4, does not exist.
Book lines move when each net's rerun lands (§2.5). The landed B0 runs trained the whole-image
fallback; B0's chapter says what ran.

Also in this change: the A3 family (`short` and its derivatives) drops its `augBicubic := false`
pin and trains C6's bicubic RandAugment geometry like A2/A1 (`imagenet_parity.md` X5); the landed
A3 pair trained bilinear. `aug_bicubic_pil_check.py` green on the A3 trainer; the two A3 rerun
confs (`r50-a3-jax-4gpu`, `r50-a3-wxclip4x128-bf16-4gpu`) refuse a file without the bicubic warp.
Not measured: producer throughput under the new sampler and resize kernels (the side-quest ETAs
predate it, as they predate X2 / X5 / X6; D17).

### 2.1 What differs

| | torchvision `RandomResizedCrop` (timm, DeiT, ConvNeXt, RSB, torchvision's R34/R50) | ours (`jax/Jax/Codegen.lean` ~501, `tf.image.sample_distorted_bounding_box`) |
|---|---|---|
| area | `U(0.08, 1.0)` of the image | `area_range=(0.08, 1.0)`, but `min_object_covered=0.1` makes the floor 10% |
| aspect ratio | log-uniform on `[3/4, 4/3]` | uniform on `[3/4, 4/3]` |
| tries | 10 draws of (area, ratio); first that fits | TF's own search, `max_attempts=10` |
| fallback | centre crop at the ratio bound the image violates | the whole image (`use_image_if_no_bounding_boxes`) |
| offsets | `i ~ U[0, H−h]`, `j ~ U[0, W−w]` | TF's |
| resize | PIL, `interpolation` per recipe (DeiT, ConvNeXt `bicubic`; timm `train.py` default `random` = bilinear or bicubic per image) | `tf.image.resize(..., antialias=True)`, bicubic since C6 |

The resize kernel is a sub-item: `bicubic` is right for DeiT and ConvNeXt; the RSB and MNv4
recipes train under timm's `random`. Decide per recipe when the flag is set (§5).

### 2.2 Which recipes it applies to

Only the torchvision-side recipes: **R34** (2018, PyTorch examples), **R50** (2018, A3, and the
A2/A1 side quests), **MNv4** (timm), **ConvNeXt** T/S/B, **ViT** Ti/S/B. **Not MNv2 and not B0**:
TF-slim's and the EfficientNet TPU code's `distorted_bounding_box_crop` is
`sample_distorted_bounding_box(min_object_covered=0.1, aspect_ratio_range=(3/4, 4/3),
area_range=(0.08, 1), max_attempts=10)`, i.e. what we emit today, except that EfficientNet falls
back to a centre crop (`_decode_and_center_crop`) where we take the whole image — a separate,
smaller item (§5). Imagenette is untouched: its augmentation is Lean-side
(`F32.randomCrop` 256 → 224 + flip, `Verified/Train.lean` ~1150), not the tf.data pipeline.

So this is a per-recipe flag, `TrainConfig.cropTorchvision`, on in the R34 / R50 / MNv4 / ConvNeXt
/ ViT ImageNet configs and off in MNv2's and B0's — the same shape as `augBicubic` and
`erasingPixel` (C6), which is the pattern the confs' prechecks already know how to assert.

### 2.3 Implementation

In the pipeline template (`Codegen.lean`, the `_imagenet_decode_random_crop_flip` block), a
`_torchvision_rrc(shape)` emitted when the flag is set, replacing the `sample_distorted_bounding_box`
call. Vectorised, not a Python loop: draw 10 `(area, log_ratio)` pairs at once with
`tf.random.uniform(..., seed=_AUG_SEED)` (the existing op-level seed, so the shim's determinism
contract holds), `w = round(sqrt(area · r))`, `h = round(sqrt(area / r))`, `valid = (0 < w ≤ W) ∧
(0 < h ≤ H)`, take the first valid index (`tf.argmax` over the boolean row), else torchvision's
fallback (`W/H < 3/4` → `w = W, h = round(W / (3/4))`; `> 4/3` → `h = H, w = round(H · 4/3)`; else
the whole image); offsets `i ~ U[0, H−h]`, `j ~ U[0, W−w]` as integers; window `[i, j, h, w]` into
`tf.io.decode_and_crop_jpeg`. About 30 lines of emitted Python. The trainer and the shim get it
from the one template, so the pipeline stays byte-identical between the two paths
(`vit_parity_todo.md` §1 P2's invariant).

### 2.4 Checks

* `scripts/gates/crop_sampler_gate.py` (new): import the emitted function from a generated shim,
  run it 200k times at three image shapes (square, 3:4, 16:9); run torchvision's
  `RandomResizedCrop.get_params` in `.venv-timm` at the same shapes; compare fallback rate and the
  Kolmogorov distance of the `scale` and `log ratio` samples (tolerance 0.01 at that n), and the
  offset means. A `--break` control (uniform ratio, or the whole-image fallback) must go red. The
  shape of `aug_bicubic_pil_check.py`.
* `scripts/parity/identity.py` digests of the streamed wire re-pinned for the flagged nets;
  `mixup_gate.py` and `bce_target_gate.py` are downstream of the crop and keep their
  known-answer checks, re-pinned likewise.
* `scripts/regen_jax_generated.sh` (every shim and trainer moves); each flagged conf's precheck
  asserts the flag's line in the emitted trainer, as f19d859a did for the threshold.

### 2.5 Book and plans

ch 5 A2/A1 ledger (~7098) and §5.9's still-out line; ch 8 (~11006) and §8.6's still-out; ch 9
item 4 (~13311) and §9.7's still-out; appendix A (~17779). Each line moves from "does not match"
to "matches" when that net's rerun lands, not before — the book says what ran. `imagenet_parity.md`
F2 and `vit_parity_todo.md` P-G point here. B0's chapter (~9707) keeps TF's sampler as the paper's.

### 2.6 Reruns

None beyond those already owed, provided the flag lands before they launch (§1): the R34 and R50
reruns, A2/A1, MNv4 `full`, ConvNeXt-S/B, ViT-S/B. The landed MNv4 Conv-M pair, ConvNeXt-T pair
and ViT-Ti pair would not be rerun for the crop alone; ViT-Ti and ConvNeXt-T fold into the GELU
rerun (§3.9), MNv4 Conv-M stays as a disclosed line.

**Cost: about one day** (template + gate + regen + digests + prechecks), before any run.

## 3. Item 3: the exact-erf GELU

**Status 2026-10-06: §3.3–§3.6 committed (0ce30dc7, 10fc13ee); §3.7's JAX flag in; no committed
render carries the exact form yet.** Begin at §3.8.

### 3.1 Today

The tanh approximation on both paths. Lean: `geluScalar`, `geluScalarDeriv_eq`, `geluHasVJP`
(`Proofs/Architectures/Activations.lean`); SHlo ops `geluF` / `geluBack` and the batched
`gelu` / `geluBackB` (`Codegen/StableHLO/Basic.lean` ~232, ~723, ~841), printed as arithmetic plus
`stablehlo.tanh` (`Pretty.lean` ~2528), with parser tokens (`Parse.lean` ~152), the backward bridge
`gelu_back_bridge` (`Foundation/IR.lean` ~371) and the forward faithfulness theorem (`Basic.lean`
~3615); about twenty ViT and ConvNeXt proof files name `gelu` / `geluBack` in their chains
(`ViTBackB0`, `ViTStepTie`, `ViTMhsaBackCertifiedTie`, `ConvNeXtBackB0`, `ConvNeXtStepTie`, …).
JAX: `jax.nn.gelu(x)` — `approximate=True` — at three sites in `Codegen.lean` (~1387, ~1414, ~2338).
The parity gates pin tanh and use erf as their red control (`vit_timm_parity.py --controls`,
`cnx_timm_parity.py`): tanh against erf is about 4e-5 of logit scale at init, so the CNN gates'
1e-3 would not see it and the ViT gate runs at 1e-5. The float tier (`Proofs/Float/`) carries no
transcendental op, so neither tanh nor erf enters a bridge bound.

### 3.2 Target

`gelu(x) = x · Φ(x)`, `Φ(x) = ½ (1 + erf(x/√2)) = ½ erfc(−x/√2)`, derivative `Φ(x) + x · φ(x)`
with `φ(x) = exp(−x²/2) / √(2π)`. Emitted in the erfc form, as JAX spells it (§3.3).

### 3.3 Step 0: the op exists downstream? Yes, and the op is `chlo.erfc`

**Done 2026-10-06 on ares** (RTX 4060 Ti, the pinned `jax-cuda12` plugin 0.11.0, PJRT API 0.114;
IREE 3.12.0rc20260428). StableHLO has no erf; CHLO has `chlo.erf` and `chlo.erfc`, which JAX's
`lax.erf` / `lax.erfc` lower to. One-op modules for each and a forward-plus-backward GELU module,
65,536 points on [−40, 40], went through a fresh build of `ffi/pjrt_ffi.c` and through
`iree-compile` (`cuda` at `sm_86` and `llvm-cpu`, the repo's flags, input type auto-detected),
executed by `ffi/libiree_ffi.so`. Every module compiles and executes on both, so the fallback —
XLA's rational approximation emitted as arithmetic — is not needed. The text form is the one JAX
prints: `%0 = chlo.erfc %x : tensor<…xf32> -> tensor<…xf32>`.

**The spelling.** JAX 0.11's `jax.nn.gelu(approximate=False)` is `(0.5·x) · erfc((−x)·√½)`, not
`1 + erf`. Emitted in JAX's op order — that forward, and the backward as `jax.vjp` prints it,
`0.5·(dy·erfc(z)) − (−2/√π)·((0.5·x)·dy)·exp(−z²)·√½` with `z = (−x)·√½` — the shim's output equals
JAX's bit for bit at all 65,536 points, forward and backward. The `1 + erf` spelling differs from
JAX at 47% of the points (by up to 9.5e-7) and loses the negative tail: 6% relative error at
x = −5 and exactly 0 below −5.24, where the erfc form holds 4e-8. PyTorch's `nn.GELU` (2.13, CPU)
is the `1 + erf` form over its own erf kernel, so neither spelling reproduces it bit for bit; it
carries the same tail loss (0 below −5.54). So §3.4 and §3.5 are written for `chlo.erfc`.

| max abs difference, erfc spelling, unit cotangent | forward | backward |
|---|---|---|
| PJRT CUDA against the float64 closed form | 3.7e-7 | 1.4e-7 |
| IREE against float64 (`cuda` and `llvm-cpu` agree) | 3.7e-7 | 1.4e-7 |
| IREE against PJRT (what the differential oracle sees; `grad_tie.py` runs at 2e-3) | 2.4e-7 | 1.2e-7 |
| PJRT against `jax.nn.gelu` / `jax.vjp`, same card | 0 | 0 |
| PJRT against PyTorch `nn.GELU` and its autograd | 4.8e-7 | 2.4e-7 |
| PyTorch against float64 | 3.7e-7 | 2.5e-7 |
| today's tanh form against float64 | 4.7e-4 | 8.7e-4 |

**Cost and dtype.** One GELU on the card costs the same either way: tanh → erfc is 0.757 → 0.762 ms
forward and 1.378 → 1.393 ms forward-plus-VJP at ViT-Ti's 128×197×768, 2.670 → 2.697 and
5.185 → 5.132 ms at ViT-B's 128×197×3072 (JAX, medians of 200, the same kernels the render gets).
Both paths run the GELU in f32 — the bf16 renders convert after it and JAX's `mm` returns f32 — so
no bf16 kernel is involved.

**Not asked.** The 3060 box's `xla_cuda13` plugin, where §3.9's reruns go, and the Orin's build. A
`chlo.erfc` fixture in the platform suite's tier 0 asks every platform; it goes in with §3.8. The
probe itself was a scratch run and is not in the repo.

### 3.4 Proofs: done

**Done 2026-10-06, committed 0ce30dc7.** Two leaf modules, 264 lines, nothing downstream rebuilt:
`Proofs/Architectures/GeluErf.lean` and `GeluErfGaussian.lean`. `GeluErfGaussian` is a `Certs`
root; five declarations are in `tests/AuditAxioms.lean` (1,813 verdicts, all within the three
axioms); `lake build Certs`, `docstring-checkrefs`, `name_lint.py`, `module_refs.py`,
`check_audit_coverage.py` and `import_audit.py implied` are green. Mathlib (v4.34.0) has no error
function.

* `GeluErf.lean`, on `Tensor` and the interval-integral FTC: `gaussPdf x = exp(−x²/2)/√(2π)` and
  `gaussPhi x = ½ + ∫₀ˣ gaussPdf`; `hasDerivAt_gaussPhi` (`Continuous.integral_hasStrictDerivAt`);
  `geluErfScalar x = x · gaussPhi x`, `geluErf`, `geluErfScalarDeriv` (a `deriv`, as
  `geluScalarDeriv` is), `geluErfScalarDeriv_eq` (`Φ + x · φ`), `pdiv_geluErf`, `geluErfHasVJP`,
  `geluErfHasVJP_correct`, and the `fun_prop` differentiability lemmas; `erf`, `erfc`, `erf_neg`,
  `gaussPhi_eq_erf`, `gaussPhi_eq_erfc`; and the two forms as §3.3 computes them,
  `geluErfScalar_eq_erfc : geluErfScalar x = 0.5 · x · erfc(−x · √½)` and
  `geluErfScalarDeriv_eq_erfc : … = 0.5 · erfc(z) + (2/√π) · (0.5 · x) · exp(−z²) · √½`, `z = −x · √½`.
* `GeluErfGaussian.lean`, on Mathlib's Gaussian and CDF: `gaussPdf_eq_gaussianPDFReal`,
  `integral_Iic_zero_gaussPdf` (the `½`), `gaussPhi_eq_measureReal_Iic` and
  `gaussPhi_eq_cdf : gaussPhi x = cdf (gaussianReal 0 1) x`.

Two departures from the sketch. The density is a closed form in the core, not `gaussianPDFReal`:
§3.5 makes `StableHLO/Basic.lean` import the core, and its 265 downstream modules then load 438
more Mathlib modules for the interval integral, against 869 with the probability library; the
equality with `gaussianPDFReal 0 1` is the leaf's first theorem. And `Φ` is proved to be Mathlib's
CDF of the standard normal, which the sketch left out and which is what makes the `½` in the
definition a theorem.

Parked for §3.5's rebuild: `Activations.lean`'s module docstring says the tanh form is "the
function here and in the renders". A pointer to `GeluErf` there rebuilds `LayerNorm` and the 279
modules above `Activations`, so it goes in with the AST change.

### 3.5 AST, printer, parser, gates: done

**Done 2026-10-06, committed 10fc13ee.** The four GELU constructors carry the form, `GeluForm` (`.tanh` or
`.erf`, §3.6), rather than doubling: a first cut added `geluErfF` / `geluErfBack` / `geluErf` /
`geluErfBackB` beside the tanh ones, and §3.6 folded them back, because a selector over two
constructors makes `den` case on the form and a parameter does not. Every committed render is
the `.tanh` instance and its bytes are unchanged.

* `StableHLO/Basic.lean`: `geluF gf` / `geluBack gf`, the descriptor `BatchableOp.gelu gf` and
  `geluBackB gf`; their `den` arms are `gf.map` / `gf.hasVJP`, with no case split;
  `geluF_faithful`, `geluBack_faithful` and `den_geluBackB` hold at either form by `rfl`.
  `Foundation/IR.lean`: `geluErf_back_bridge`, and the one import of `GeluForm`.
* `Pretty.lean`: `Raw` and `Tok` carry the form; `skel` gives the two forms different batched
  tags (`"gelu"` / `"geluErf"`, `"geluBackP"` / `"geluErfBackP"`); the tanh emit is untouched, and
  the exact emit is one text function per direction — `geluErfFwdText`, `geluErfBackText` —
  shared by the token and its batched tag, in JAX's op order (§3.3). `Parse.lean`: `parseStack`.
* The text, run: `renderModule` of `.geluF .erf` and of `.geluBack .erf` through a fresh shim
  equals `jax.nn.gelu(approximate=False)` and its `jax.vjp` bit for bit at 65,536 points (random
  cotangent); IREE (`cuda` and `llvm-cpu`) within 2.4e-7 forward, 4.8e-7 backward.
* `tests/TestBatchedEmitTie.lean`: the two batched forms tie their per-example peers (51 forms),
  and a pair check: one `chlo.erfc` at √½, no tanh and no `chlo.erf` in the exact emit; one
  `stablehlo.tanh` and no `chlo` op in the tanh emit; the two distinguishable.
* `IRPrint.lean` and `check_ir_codegen.py`: `gelu_erf_fwd` / `gelu_erf_back` modules, ALL PASS on
  IREE `llvm-cpu` (run from the sibling `lean4-jax/.venv`; this repo's `.venv` has no `iree`).
* `check_jacobians.py`, 31 of 31: `pdiv_gelu` had been checking the exact form under the tanh
  theorem's name, and skipping without scipy. It now checks the tanh closed form, the exact form
  is `pdiv_geluErf`, and neither needs scipy.
* `Activations.lean`'s docstrings point at `GeluErf` (§3.4's parked item); `tests/AuditAxioms.lean`
  gains `IR.geluErf_back_bridge`.

`lake build Certs` rebuilt the 181 modules above `Activations.lean` in 4 min 20 s of wall time
(80 CPU-minutes, 32 cores), so each of §3.6's rechecks is minutes.
Re-elaborating every renderer rewrote `verified_mlir/` with no diff. `lake build`, `Apps`,
`TestSupport`, `docstring-checkrefs`, `name_lint.py`, `module_refs.py`, `comment_numbers.py`,
`check_audit_coverage.py`, `check_render_coverage.py`, `repo_shape.py`, `import_audit.py implied`
and the five `#guard` scripts of `certs.yml` are green.

Moved out of this step: `convention_audit.py` counts `jax.nn.gelu` calls and pairs with §3.7's
flag; the comparator DECLS and `formalization.yaml` carry whole-net rows, none for an op, so
theirs come with §3.6's statements; `tests/ViTRender.lean` is §3.8's.

### 3.6 The nets and the ties: done

**Done 2026-10-06, committed 10fc13ee.** Parametrised, not duplicated; no budget had to move.

* `Proofs/Architectures/GeluForm.lean`: `inductive GeluForm | tanh | erf`, with `scalar`,
  `scalarDeriv` (a `deriv`), `map`, `pdiv_map`, `hasVJP`, `hasVJP_correct`. At each form they are
  `gelu` / `geluHasVJP` and `geluErf` / `geluErfHasVJP` by `rfl`. Only `scalar` and its
  differentiability case on the form, so a definition built on `gf.map` unfolds the same way at
  both.
* The cone, measured inside Lean (a metaprogram over the environment, type and definition
  dependencies): 263 definitions and 264 theorems in 32 modules depended on the tanh GELU — the
  ViT and ConvNeXt architecture files, back chains, parameter gradients, step ties, the four
  renderers and `SpecVJP`. Each definition now takes `(gf : GeluForm)` first, each theorem
  `{gf : GeluForm}`; about 2,100 use sites. The same metaprogram afterwards finds no declaration
  outside the GELU's own files that depends on a particular form.
* What the proofs needed: two differentiability proofs lose `geluScalar` from their `unfold`
  (`GeluForm.scalar_differentiable` is a `fun_prop` lemma), and 21 uses of a theorem with no
  expected type, a `have … := thm …`, pin `(gf := gf)`. Nothing else changed in a proof body; no
  heartbeat or kernel budget moved; the corpus rebuilds in the same 4 minutes.
* Renders: the 93 `#eval` writers of `ViTRender`, `ViTRenderB`, `ConvNeXtRender` and
  `ConvNeXtRenderB` pass `.tanh`, and `verified_mlir/` is rewritten with no diff;
  `vit-fwd-b-tie` and `convnext-fwd-b-tie` are byte-identical against the committed artifacts;
  `FwdGraphTextTies`' ConvNeXt block guard checks both forms.
* The exact form, whole net (scratch renders, §3.8 commits them): ViT-Tiny's forward carries 12
  `chlo.erfc` and its AdamW step 24, ConvNeXt-T's 18 and 36, none a `stablehlo.tanh`. All four
  compile through the shim on CUDA (the ViT step in 8.4 s, 603 outputs); both forwards compile on
  IREE.
* Statements: the comparator's architecture pair takes `gf` in its six GELU statements
  (`chk_pdiv_gelu` and `chk_geluHasVJP_correct` are now about `GeluForm.map` / `GeluForm.hasVJP`),
  and the regenerated tier quantifies six of its 54 over the form; `tests/comparator/run.sh`
  passes all three configurations. `tests/AuditAxioms.lean`: 1,816 verdicts, all within the
  three axioms.

The spec language does not carry the form. `VLayer.transformerBlock` and the ConvNeXt block
constructors are unchanged, and `SpecVJP`'s ConvNeXt-T and ViT-Tiny denotations take `gf` as an
argument: like weight-decay exclusion, clipping and drop-path, the GELU is a property of the
render variant (§3.8's tag), not of the layer list. The sketch had it in the spec.

Left as it fell: the pass pushed 183 more lines past 100 columns in files that already had 838
such lines; nothing lints line length.

### 3.7 JAX side: done

**Done 2026-10-06.**

* `TrainConfig.geluExact` (default off) and `geluPy` in `jax/Jax/Codegen.lean`: the three GELU
  sites — the transformer MLP, the ConvNeXt block, a `.gelu` activation — emit
  `jax.nn.gelu(…, approximate=False)`. Set in the six ImageNet base configs
  (`vit{Tiny,S,B}ImagenetConfig`, `convNeXt{Tiny,S,B}ImagenetConfig`; the derived recipes
  inherit). Re-emitted: 18 trainers move by their one GELU line, the other 56 files of
  `jax/generated/` are byte-identical. Imagenette's configs are untouched (§5 a).
* `vit_timm_parity.py` pins each net at what its config asks for: the ImageNet references at
  DeiT's own exact GELU and LN ε 1e-6, the Imagenette net at tanh and 1e-5. JAX against timm:
  5.7e-7 / 9.1e-7 / 1.4e-6 (Ti / S / B) and 4.7e-7 (Imagenette); the verified half renders
  `vitFwdRenderB .erf` at ε 1e-6 through IREE: 5.7e-7 / 1.1e-6 / 1.8e-6. Controls red: k/v
  swapped 1.3e-1, timm's tanh GELU 3.7e-5.
* `cnx_timm_parity.py` checks against timm's ConvNeXt as it ships: 6.5e-7 / 4.7e-7 (T at 224 /
  288), 6.4e-7 / 5.5e-7 (S), 1.1e-6 / 8.5e-7 (B). Controls red: timm's tanh GELU 1.4e-4, a
  LayerScale swap 1.1e-2. Its tolerance goes from 1e-4 to 1e-5: at 1e-4 the GELU control was red
  by a factor of 1.4. `--paper` is now the default and `--tanh` measures the approximation.
* `pc_gelu` in `scripts/lib/precheck.sh`, called by the five JAX job confs
  (`vit-default-jax`, `vits-default-jax`, `vitb-accum-jax`, `cnxs-default-jax`,
  `cnxb-default-jax`): every `jax.nn.gelu(` call in the trainer carries `approximate=False`. The
  box's pre-flag trainers were refused under `DRY_RUN`; after `regen_jax_generated.sh sync` all
  five pass. ConvNeXt-T's JAX reference has no job conf (it launched from
  `jax/scripts/supervise_convnext_t_300ep_3060.sh`), so its rerun has neither `pc_crop` nor
  `pc_gelu` until it gets one (§3.9).
* `regen_jax_generated.sh check`, `convention_audit.py --selftest`, the seven timm parity
  invocations of `jax.yml`, `shim_wiring_gate.py`, target names, repo shape, module refs, comment
  numbers and `docstring-checkrefs` are green; the default targets, `Apps` and `TestSupport`
  build.

Two things this step found in the gates.

* **The timm parity gates emitted from stale `.olean`s.** They run `lake env lean`, which never
  rebuilds. `MainVit` had not been rebuilt since `TrainConfig` gained fields (the crop commit and
  this one), and its config read back wrong: the Imagenette reference came out with LayerNorm ε
  `0.0`, and the gate passed anyway at 8.4e-6 under its 1e-5. All six gates now `lake build` the
  module they emit from first; rebuilt, the Imagenette number is the 4.7e-7 recorded on 09-29.
* **`convnext_forward_tie.py` cannot see the form of the GELU at its default weight scale.**
  With the reference at the exact GELU and `verified_mlir/convnextin_fwd.mlir` still the tanh
  render, it passes at 5.4e-7 (`--scale 0.05`: the pre-activations are too small for the two
  forms to differ). At `--scale 0.3` it fails at 4.6e-4, as it should. §3.8 points it at the
  exact render and gives it a scale and a control that see the form.

Also: `planning/rubric_review/wp2_combo_gen.py` spelled the ViT and ConvNeXt prefixes without the
form and no longer reproduced the two `*_net_tied_lossGrad` theorems after §3.6; it does again
(0 differing lines on ViT, ConvNeXt and EfficientNet).

### 3.8 Renders and the artifact gates

New variant names for the ViT and ConvNeXt ImageNet renders (an activation token in the tag — the
old names stay, they are the landed runs' artifacts), plus the Imagenette renders if §5(a) says so.
Render guard: every new `verified_mlir/` file into `proofs.yml`'s diff list
(`render-guard-on-new-artifact`); `gen_mlir_manifest.py`; `check_render_coverage.py`; the IREE
differential oracle (`vjp_oracle`) and `grad_tie.py` over the new ops; `tests/ViTRender.lean`; a
`chlo.erfc` fixture in the platform suite's tier 0 (§3.3).

From §3.6 and §3.7: the variant tag is where the form lives (the spec language does not carry
it), so the verified drivers need a `checkLnEpsWorld`-style refusal of a train step at one form
scored by a forward at the other; `convnext_forward_tie.py` moves to the exact render at a
weight scale that separates the forms, with the tanh render as its red control; the two
whole-net tie tests (`vit-fwd-b-tie`, `convnext-fwd-b-tie`) run their batched-against-per-example
backward check at `.erf` as well as `.tanh`; and the verified ImageNet job confs assert the
exact render the way the JAX confs now assert the trainer.

### 3.9 Reruns and the book

* **ViT-Ti pair**: 45.9 h (JAX) + 64.9 h (verified) on the 3060 box. **ConvNeXt-T pair**: 76.5 h +
  91.3 h — the EMA pair D15 is owed anyway, so this is the rerun it should be. Both carry the crop
  (§2) as well. About twelve days of that box.
* **ViT-S/B and ConvNeXt-S/B** carry it if it lands before they launch (§1).
* **Imagenette** ViT (§9.4, the bit-exact claim re-measured) and ConvNeXt (ch 8): hours on one
  card, if §5(a) flips them.
* Book: Definition `ax:geluScalar` becomes the erf form, with the tanh approximation as the
  remark about the runs before it; ch 8 (~11002) and ch 9 item 3 (~13309), the two still-out lists;
  `imagenet_parity.md` D1 and `vit_parity_todo.md` P-A point here.

### 3.10 Cost

| step | days |
|---|---|
| 3.3 probe | done |
| 3.4 proofs | done |
| 3.5 AST / printer / parser / gates | done |
| 3.6 ties parametrised, rechecked | done |
| 3.7 JAX flag and gates | done |
| 3.8 renders, artifact gates | ½–1 |
| 3.9 reruns | ~12 days of the 3060 box, not of a person |

## 4. Not touched

§9.6 items 1 (DeiT's batch of 1024 — hardware), 2 (the clip — `vit_parity_todo.md` P-B's 30-minute
clip-off probe is a separate decision), 5 (the per-epoch schedule and the per-batch Mixup/CutMix
switch, P-D / P-E) and 6 (the smaller things); and G2–G5 between the two arms, which are not
paper items.

## 5. Decisions

Decided 2026-10-06: (a) ImageNet ViT and ConvNeXt only; Imagenette stays. (b) The ViT-Ti and
ConvNeXt-T reruns go ahead of the side-quest queue, after the code for both items is done; no runs
before then. (c) Taken with the crop flag. (d) Per recipe: bicubic for DeiT / ConvNeXt; `random`
for RSB and MNv4; bilinear for R34 and R50-2018 (torchvision's default, the PyTorch-examples recipe).

(a) Land both behind flags and flip the ImageNet ViT and ConvNeXt defaults together
(recommended), or also flip the Imagenette chapters, which re-measures ch 8's and ch 9's chapter
numbers (hours, one card). (b) The ViT-Ti and ConvNeXt-T reruns: ahead of the side-quest queue,
or after it drains (~78 days). (c) B0's crop fallback — EfficientNet's centre crop against our
whole image — take it with the crop flag, or leave it as a disclosed line. (d) The training
resize kernel per torchvision recipe: `bicubic` (DeiT, ConvNeXt) against timm's `random`
(RSB, MNv4).
