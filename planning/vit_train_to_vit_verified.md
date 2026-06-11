# Planning — `vit-train` → `vit-verified` (recipe parity)

The follow-on to `planning/vit_close.md` (which closed the *render*) and
`planning/verified_train_step.md` (the TCB philosophy). Goal: bring the **verified**
ViT runner (`vit-verified`, `VerifiedNet.train` over `ViTRender.lean`) up to
**training-recipe parity** with the **unverified** runner (`vit-train`,
`MainVitTrain.lean` + `LeanMlir/Train.lean` Adam path) that produced the original
Imagenette ViT-Tiny number — so the proof-rendered path can reproduce that
accuracy, not just compile-and-run.

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
| augmentation | mixup/CutMix | none | host (Phase 5) |
| EMA / grad-clip | available | none | host / render (Phase 5) |
| epochs | 80 | 20 (cfg default) | host (Phase 6) |

**A transformer does not reach ~65% on 9.5k images with vanilla SGD + no
warmup + no aug.** So `vit-verified` as-is (after the loader fix) gives a
*legitimately verified* training curve that lands well below the target, and the
entire gap is recipe. The **load-bearing missing piece is a verified Adam step** —
and Adam is also the one item the existing SGD descent suite
(`sgdW_descends_certified_grad`, `linear_sgd_descends`, the CNN rungs) does *not*
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
  just which `f32` we feed), the data pipeline (CutMix/mixup/crop), EMA weight
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
`StableHLO.sgdW_descends_certified_grad` / the `vit_render_*_chain_certified`
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

## Phase 5 — Augmentation (CutMix/mixup), EMA, grad-clip

Mostly **host-side**, riding on Phase 4's soft-label path:
- **CutMix/mixup (H):** patch/blend two images and produce the mixed soft label
  (`λ·t_a + (1−λ)·t_b`) → straight into the soft-label CE. Your own ablation flags
  CutMix as *the* load-bearing accuracy knob at 9.5k images, so this is where the
  number actually moves. No new theorem (the label is just an input).
- **EMA (H):** maintain an exponential moving average of `params` on the host for
  eval; pure post-hoc averaging, outside the proof boundary.
- **Grad-clip (decision):** global-norm clipping scales every gradient by
  `min(1, c/‖g‖)` *before* the update — i.e. inside the fused train-step. Either
  (a) render it as a token (extra proof: the scaled gradient still denotes
  `clipScale · certified-grad`; `emitAdamUpdate` already takes a `clipScale`), or
  (b) skip it for ViT-Tiny (warmup usually suffices). Recommend (b) first; (a) if
  early-epoch instability appears.

Acceptance: CutMix run materially closes the gap to the target.

## Phase 6 — Scale + match the number

Bump `VerifiedConfig` to 80 epochs (the `vitTinyConfig` budget), run the full
Adam + cosine + warmup + wd + label-smoothing + CutMix recipe through the verified
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
| SGD-step certification (the pattern to mirror) | `StableHLO.sgdW_descends_certified_grad` |
| CE head proof to generalize | `lossCot_eq_softmax_sub_onehot`, `softmaxCE_grad` |
| axiom audit (add new theorems here) | `tests/AuditAxioms.lean` |
| loader-free GPU parity check | `scripts/render_parity.py` |
| prior context | `planning/vit_close.md`, `planning/verified_vit.md`, `planning/verified_train_step.md` |
