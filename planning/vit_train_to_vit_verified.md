# Planning — `vit-train` → `vit-verified` (recipe parity)

The follow-on to `planning/vit_close.md` (which closed the *render*) and
`planning/verified_train_step.md` (the TCB philosophy). Goal: bring the **verified**
ViT runner (`vit-verified`, `VerifiedNet.train` over `ViTRender.lean`) up to
**training-recipe parity** with the **unverified** runner (`vit-train`,
`MainVitTrain.lean` + `LeanMlir/Train.lean` Adam path) that produced the original
Imagenette ViT-Tiny number — so the proof-rendered path can reproduce that
accuracy, not just compile-and-run.

---

## STATUS (2026-06-11) — DONE through recipe parity + differential-tested

The `vit-verified-adam` exe (`MainViTVerifiedAdam`, `lean_exe vit-verified-adam`) now
trains ViT-Tiny through the verified render with **full recipe parity** to
`vitTinyConfig`, and has been **differential-tested against `vit-train` on identical
256² Imagenette** — it tracks the reference step-for-step.

**Implemented (commits `c939ea2` → `c8c66ac`):**
- **Phase 0** loader fix — `train.bin` 224² vs the loader's 256² assumption;
  `VerifiedTrain.loadData` defaults to 224²/no-crop, `LEAN_MLIR_IMAGENETTE_TRAIN=256`
  for the canonical 256²-train + random-crop split.
- **Phase 3a** ℝ AdamW spec (`Proofs/AdamStep.lean`: `adamWParam` etc.) — audited (531/531 3-axiom).
- **Phase 3b** AdamW render (`ViTRender.emitAdamV`) + den-level faithfulness
  (`Proofs/AdamRender.lean`: `adamW_certified_grad`, linear-net) + GPU numeric pin (0.49).
- **Phase 3c** packed `[θ|m|v]` threading through the generic FFI (no `.so` change).
- **Phase 2** runtime lr + cosine + warmup + exact bias correction (lr/bc₁/bc₂ ride as
  rank-0 scalar params in the blob tail; `vitTrainStepModuleAdamSched` / `trainAdamSched`).
- **Phase 4 render** label smoothing in-graph (soft-target CE cotangent); proof rung pending.
- **Phase 5** data-pipeline augmentation: random crop 256→224 + random hflip (`F32.randomCrop`/`randomHFlip`).
- **Recipe-match** to `vitTinyConfig`: lr 3e-4, 80 ep, wd 1e-4, cosine, warmup 5, label-smooth 0.1,
  augment; EMA + grad-clip correctly OMITTED (vitTinyConfig sets neither).
- **Differential-test plumbing**: per-epoch `F32.shuffle` (a real bug the diff-test caught —
  class-sorted data made single-class batches, val_acc stuck at the 9.4% random floor); in-graph
  smoothed-CE loss surfaced to the (unread) lr output slot → per-step/epoch loss logging matching
  vit-train; auto checkpoint/resume (`.lake/build/<slug>_adam_ckpt.bin{,.epoch}`).

**Differential-test result (256² Imagenette, identical recipe):** step 100 verified `2.111` vs
reference `2.124`; epoch 1 loss `2.10` vs `2.11`, lr identical; **val_acc 39.7%→67.4% over 25
epochs** (healthy curve, tracking the reference) before an external reap. The proven gradient
drives the same learning as the trusted one.

**To run (clean 80-epoch):** `rm -f .lake/build/vit_adam_ckpt.bin*` first (else it resumes from the
last checkpoint — currently epoch 1), then
`HIP_VISIBLE_DEVICES=0 LEAN_MLIR_IMAGENETTE_TRAIN=256 .lake/build/bin/vit-verified-adam <dir-with-/imagenette>`
(256² split is at `claude_max/lean4-jax/data`). ~6–7h (verified is ~2× slower/step). A watchdog
loop that auto-resumes on reap is the robust way to get the full 80 epochs.

**Remaining (not blocking the run):**
1. **Finish the 80-epoch run** for the final top-1 (vs the reference's number).
2. **~2× per-step overhead** — the `[θ|m|v]` blob rebuild (concat+extract, 2×66 MB/step). Optimize by
   reusing the output blob + overwriting the 12-byte scalar tail in place (needs an f32-write into a
   ByteArray). Halves the run time; benefits every verified trainer.
3. **Verification-depth rungs** (independent of the number): the ViT-specific Adam den-close
   (`vit_render_*_adam_certified`, the linear-net `adamW_certified_grad` lifted to ViT), the
   soft-target CE gradient proof (`softmaxCE_grad_softTarget`), and the text↔den extraction.

---

## Headline: the gap is the OPTIMIZER + the RECIPE, not the model.

The verified render already emits the **real ViT-Tiny** and it is proof-faithful:
vector LayerNorm (`g*_i : tensor<192xf32>`), multi-head attention (3 heads), 12
distinct-param blocks, patch-16 embed, CLS+pos, 200 params — covered by
`vitFwdGraphKMHV_faithful` and the `vit_render_*_chain_certified` param closes,
all 3-axiom clean. It iree-compiles and runs on GPU today (200/200 finite,
non-zero updates via `scripts/render_parity.py --fn vit_train_step`).

What's missing is everything *around* the gradient. `ViTRender.vitTrainStepModule`
bakes the learning rate as a **constant** (`stablehlo.constant dense<0.1>`) and
emits a pure SGD update (`θn = θ − grad·lr`) against a **hard-label**
`softmax − onehot` head. The original number came from `vitTinyConfig`:

| knob | `vit-train` (`vitTinyConfig`) | `vit-verified` today | where the fix lives |
|------|------------------------------|----------------------|----------------------|
| optimizer | **Adam(W)** (m/v moments) | plain SGD | **render + proof** (Phase 3) |
| weight decay | 1e-4 | 0 | render + proof (Phase 3) |
| LR schedule | cosine decay | constant | render (lr→arg) + host (Phase 2) |
| warmup | 5 epochs | none | host (Phase 2) |
| label smoothing | 0.1 | none (one-hot) | render (≈free) + proof (Phase 4) |
| augmentation | random crop + hflip (CutMix/mixup = ch9/10) | none | host (Phase 5) |
| EMA / grad-clip | available | none | host / render (Phase 5) |
| epochs | 80 | 20 (cfg default) | host (Phase 6) |

**A transformer does not reach ~65% on 9.5k images with vanilla SGD + no
warmup + no aug.** So `vit-verified` as-is (after the loader fix) gives a
*legitimately verified* training curve that lands well below the target, and the
entire gap is recipe. The **load-bearing missing piece is a verified Adam step** —
and Adam is also the one item the existing SGD descent suite
(`sgdW_isCertifiedGradStep`, `linear_sgd_descends`, the CNN rungs) does *not*
cover. This doc is mostly the plan for rendering and proving it.

---

## What "verified" can and can't mean per feature (the proof/host boundary)

"Verified version of `vit-train`" does **not** mean proving CutMix. It means: the
**gradient and the parameter-update arithmetic** are a proven function of the
certified ℝ gradient (3-axiom-audited), and everything that is genuinely a
*scalar schedule* or *data choice* is a documented **host-side input** outside the
proof boundary — same discipline as `planning/verified_train_step.md`'s TCB.

- **(P) Proof-rendered + audited** — the gradient (already done) and the optimizer
  *update map* (Adam/AdamW, weight decay): `θ' = adamStep(θ, m, v, ĝ, …)` where
  `ĝ` is the certified `vit_render_*_chain_certified` gradient. Faithfulness only.
- **(H) Host-side, named in the TCB** — the lr *value* each step (cosine/warmup is
  just which `f32` we feed), the data pipeline (random crop / hflip / mixup), EMA weight
  averaging, epoch count. These are inputs to a proven function, not part of it.

The TCB stays: `iree-compile`, Float32 arithmetic, the FFI. Note `FloatBridge`
covers linear/MLP only — **ViT has no Float32 budget** (no LayerNorm/softmax/exp
rounding bounds, the `exp`-accuracy rung is unproven), so Adam's float behavior is
*not* budgeted either. The verified claim is ℝ-faithfulness, as everywhere else.

**No descent theorem for Adam.** The SGD path proved both faithfulness *and*
`sgd_descends` (loss provably drops). Adam has **no** clean per-step descent
guarantee (it isn't monotone; cf. Reddi et al. 2018, the AMSGrad
counterexample). So the verified-Adam target is **faithfulness only** — the
rendered update equals the real Adam update of the certified gradient — and we
state explicitly that no descent/convergence property is claimed. This is the
honest scope, and it's the right one: it's what's true and what's provable.

---

## Phase 0 — Loader fix (prerequisite; unblocks any real run)

`VerifiedTrain.lean:130` calls `F32.loadImagenetteSized (train.bin) 256`, but the
`train.bin` in this data dir is stored at **224²**, not 256² (root-caused by exact
arithmetic: `1425359105 = 4-byte header + 9469 × (1 label + 224·224·3 uint8)`,
9469 = Imagenette train count; val is the same format and loads fine only because
`loadImagenette` hardcodes 224). Record-size mismatch ⇒ `short read`.

**Fix:** in `loadData`'s `.imagenette` branch, load train like val —
`loadImagenetteSized (train.bin) 224` (or `loadImagenette`), `crop? := false`,
`trainPix := 3·224·224`. Acceptance: `mnist`/`cifar` unaffected; `vit-verified`
reaches the train loop without exception. (Independent of everything below; do it
first so each later phase can be smoke-trained.)

## Phase 1 — Verified ViT trains end-to-end (SGD baseline; honest floor)

No new theorems. After Phase 0, `lake build vit-verified` and run it as-is to get
the **plain-SGD verified-codegen** training curve on Imagenette. Purpose: a
real, reproducible floor and a regression anchor for the phases below. Expect it
to underperform 65% substantially — that *is* the result that motivates Adam.

Acceptance: a full multi-epoch run, monotone-ish train loss, a logged val number,
exit 0. Record it in `RESULTS.md` as "verified, SGD, no-aug" so the recipe deltas
are legible.

## Phase 2 — Runtime LR + cosine + warmup

Today `lr` is a baked `constant`. Two changes:
1. **Render (P-adjacent, trivial):** add an `%lr : tensor<f32>` argument to
   `@vit_train_step` and broadcast it into each `{nm}_st = multiply grad, lr`
   instead of emitting `{nm}_lr = constant dense<…>`. Pure plumbing; the
   `*_certified` theorems are unchanged in shape (`θ − lr·ĝ` with `lr` now a free
   variable — arguably *more* general).
2. **Host (H):** `VerifiedNet.train` computes `lr = schedule(epoch, step)` with
   cosine decay + `warmupEpochs` linear warmup (port the `baseLR`/`warmup`/cosine
   logic from `Train.lean`) and feeds it as the new scalar input each step.

Acceptance: render-parity (`--fn vit_train_step`) still passes; a warmup+cosine
SGD run is at least as good as Phase 1 (warmup alone usually helps ViT).

## Phase 3 — Verified Adam(W) — the load-bearing rung

The real work. Three sub-steps:

**3a. Math (`ℝ`).** Define `adamStep` in `LeanMlir/Proofs/` (peer of the SGD step):
given `θ, m, v : Vec n`, gradient `g`, hyperparameters `(β1, β2, ε, lr, wd)`, and
bias-correction scalars `(bc1, bc2)` (= `1−β1^t`, `1−β2^t`, passed in so no `pow`
in-graph),
```
m'  = β1·m + (1−β1)·g
v'  = β2·v + (1−β2)·g²
θ'  = θ − lr·( (m'/bc1) / (sqrt(v'/bc2) + ε) + wd·θ )      -- AdamW (decoupled)
```
returning `(θ', m', v')`. Prove the elementary facts you need (e.g. the update is
well-defined; `sqrt(v'/bc2)+ε > 0`). **No descent lemma** (see boundary note).

**3b. Render + faithfulness (P).** Add an Adam update token to the verified
renderer (mirror `MlirCodegen.emitAdamUpdate` at `MlirCodegen.lean:3546`, which
already does weight decay + grad-clip + runtime-lr — the reference, but it lives
on the *unverified* path). Prove the analogue of
`StableHLO.sgdW_isCertifiedGradStep` / the `vit_render_*_chain_certified`
family: the rendered `θ'` (and `m'`, `v'`) outputs **denote** `adamStep` applied
to the *certified chain gradient* `vit_render_*_chain_certified`. Name suggestion:
`StableHLO.adamUpdate_faithful` + per-param `vit_render_*_adam_certified`. Add
each to `tests/AuditAxioms.lean` (must close 3-axiom).

**3c. I/O + driver (H + plumbing).** The train-step signature grows: per param,
take `(θ, m, v)` and return `(θ', m', v')`, plus scalar inputs `lr, bc1, bc2`
(and constants `β1,β2,ε,wd` or bake them). For 200 params that's ~600 in / ~600
out — either widen the signature or pack `m`/`v` into the existing
packed-params `ByteArray` (cleaner; the driver already concatenates param slabs).
`VerifiedNet.train` allocates `m, v` zero-buffers (cf. `Train.lean`'s
`adamM`/`adamV`), threads them, and increments the timestep for `bc1/bc2`.

Acceptance: (i) `adamUpdate_faithful` + all `*_adam_certified` audit 3-axiom; (ii)
**render-parity of the verified Adam train-step against the unverified
`emitAdamUpdate` IR** on identical inputs (`render_parity.py --ref … --cand …`,
bit/ULP-level) — the strongest check that the two paths compute the same Adam;
(iii) a verified-Adam run beats the Phase-1/2 SGD floor.

## Phase 4 — Soft-label CE (label smoothing; also unblocks Phase 5)

The head's `cot` block computes `softmax − onehot` (mean over batch). The CE
gradient wrt logits for *any* target distribution `t` is `softmax − t`, so the
**render is essentially free**: feed a soft target tensor in place of `%onehot`
and the existing subtract is correct. The **proof** needs generalizing:
`lossCot_eq_softmax_sub_onehot` / `softmaxCE_grad` are stated for one-hot;
generalize to `softmaxCE_grad_softTarget : ∂CE(softmax z, t)/∂z = softmax z − t`
for a probability vector `t` (mild). Host builds the smoothed target
`(1−α)·onehot + α/K`.

Acceptance: `softmaxCE_grad_softTarget` audits 3-axiom; render-parity holds with a
soft-label input; a label-smoothed run trains.

## Phase 5 — Augmentation (random crop + hflip), EMA, grad-clip

The **baseline `vit-train` Imagenette pipeline** is plain geometric augmentation,
in the *data pipeline*, not the network: **random crop + random horizontal flip**
(hflip only on CIFAR; nothing on MNIST). That is what parity with the original
number needs — all host-side, no soft labels required:
- **Random crop + hflip (H):** per-batch geometric transforms on the input bytes
  before the forward. Random crop needs a larger source (the canonical 256² train
  split → 224, via `LEAN_MLIR_IMAGENETTE_TRAIN=256`); hflip works on the 224² data
  directly. No theorem — it only touches the input `x`.
- **EMA (H):** maintain an exponential moving average of `params` on the host for
  eval; pure post-hoc averaging, outside the proof boundary.
- **Grad-clip (decision):** global-norm clipping scales every gradient by
  `min(1, c/‖g‖)` *before* the update — i.e. inside the fused train-step. Either
  (a) render it as a token (extra proof: the scaled gradient still denotes
  `clipScale · certified-grad`; `emitAdamUpdate` already takes a `clipScale`), or
  (b) skip it for ViT-Tiny (warmup usually suffices). Recommend (b) first; (a) if
  early-epoch instability appears.

**CutMix/mixup are NOT in the baseline recipe** — they are the **Chapter 9/10
data-augmentation explorations** (the ch10 ViT ablation where CutMix is the
load-bearing knob at 9.5k images). They ride on Phase 4's soft-label path and are
a *beyond-parity* accuracy lever, not a parity requirement.

Acceptance: random crop + hflip reaches the baseline `vit-train` accuracy band.

## Phase 6 — Scale + match the number

Bump `VerifiedConfig` to 80 epochs (the `vitTinyConfig` budget), run the full
Adam + cosine + warmup + wd + label-smoothing + crop/hflip recipe through the verified
renderer, and compare the val top-1 to the original `vit-train` number. Stretch
goal: match it; honest floor: a *fully verified-gradient* ViT that trains to a
respectable, reported number, with the schedule/aug named as host-side TCB.

Acceptance: `RESULTS.md` row "ViT-Tiny, verified codegen, full recipe" within
striking distance of the unverified runner; the delta (if any) attributed.

---

## Definition of done

`vit-verified` reaches recipe parity with `vit-train` such that:
1. The **gradient** is the certified chain gradient (done: `vit_render_*_chain_certified`).
2. The **parameter update** is a proven function of that gradient —
   `adamUpdate_faithful` + `vit_render_*_adam_certified`, all 3-axiom-audited,
   render-parity-checked vs the unverified Adam IR.
3. The **soft-label CE gradient** is proven (`softmaxCE_grad_softTarget`).
4. Schedule, augmentation, EMA, epoch count are host-side and **named in the TCB**
   alongside `iree-compile`/Float32/FFI.
5. A full-recipe run produces a reported Imagenette val number.

## Risks / open questions

- **Adam ≠ descent.** State plainly that we prove faithfulness, not convergence.
  This is the cleanest place in the repo to be explicit that "verified training"
  means "verified *gradient + update*," not "verified *that it converges*."
- **No ViT Float32 budget.** `FloatBridge` doesn't cover LN/softmax/exp; the Adam
  step adds a `sqrt`/`divide` whose rounding is also unbudgeted. Out of scope here;
  flag it so the claim isn't over-read.
- **Signature blow-up.** 200 params × (θ,m,v) is a large fused function; prefer
  packing m/v into the params `ByteArray` over a 600-arg signature. Watch
  `iree-compile` time/memory on the depth-12 graph.
- **Performance.** 80-epoch ViT-Tiny on gfx1100 is a multi-hour run; budget it,
  use the 2-GPU box, and lean on `LEAN_MLIR_START_STEP` checkpoint resume.
- **Grad-clip render proof** is the one optional item that, if needed, adds a real
  token + theorem — keep it deferred unless instability forces it.

## File / symbol map

| concern | file / symbol |
|---|---|
| verified train-step render (SGD, lr-const) | `LeanMlir/ViTRender.lean` `vitTrainStepModule` |
| verified host loop | `LeanMlir/VerifiedTrain.lean` `VerifiedNet.train` (+ loader `loadData` :130) |
| reference Adam emitter (unverified) | `LeanMlir/MlirCodegen.lean:3546` `emitAdamUpdate` |
| unverified recipe / target knobs | `MainVitTrain.lean` `vitTinyConfig`; `LeanMlir/Train.lean` Adam loop |
| certified ViT gradient (done) | `vit_render_*_chain_certified`, `vitFwdGraphKMHV_faithful` |
| SGD-step certification (the pattern to mirror) | `StableHLO.sgdW_isCertifiedGradStep` |
| CE head proof to generalize | `lossCot_eq_softmax_sub_onehot`, `softmaxCE_grad` |
| axiom audit (add new theorems here) | `tests/AuditAxioms.lean` |
| loader-free GPU parity check | `scripts/render_parity.py` |
| prior context | `planning/vit_close.md`, `planning/archive/verified_vit.md`, `planning/verified_train_step.md` |
