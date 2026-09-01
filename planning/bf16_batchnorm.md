# bf16_batchnorm.md — the BatchNorm bf16 twin, and finishing Chapter 4 with it

**Opened 2026-09-01**, at the end of the session that converted Chapter 4 to a constant learning
rate and found that its precision lever cannot answer the question it asks.

**This is self-contained.** A session that has read nothing else can execute it.

▶ **This document is one session's work, and it ends in the blueprint.** Build the BN bf16 twin
(§3 steps 1-5), then use it to finish Chapter 4 (§3 step 6 and §7). The op work is not the
deliverable; the chapter is. If the twin turns out to be a week rather than a day, §7 says what
to write instead and the chapter still ships.

⚠ **Chapter 4 is IN PROGRESS and its state matters to you.** Everything in §6 below is on the
branch; read `git log` for `docs(ch4)` before editing `content.tex`.

---

## 0. Why, in one block

Chapter 4's §4.5 wants to say **"the normalized net trains the same in bf16, so Chapter 5 can use
it."** Every net from Chapter 5 onward trains in bf16; the book never checks that this is free.

⛔ **It cannot say it today, and the reason is structural.** The renderer has two convolution op
families. bf16 twins exist only for the **batched** family. `bnPerChannelF` / `bnPerChannelBack`
have **no bf16 twin in either**. So bf16 on CIFAR can only be measured on the *un-normalized* net
— the one that NaNs under AdamW at fp32 in 3 runs of 5 — which is the hardest case and the wrong
one for licensing Chapter 5.

▶ What §4.5 currently says instead, and it is measured and true: changing the **op family** at
fixed fp32 moves the answer more than dropping to bf16 does (+2.0 SGD, +1.7 momentum, against
−1.4 and −0.6). That is a real finding and it stays. It is not the claim Chapter 5 needs.

---

## 1. The exact gap

`cifar8BnTrainStepFaithfulV` (`LeanMlir/Proofs/Codegen/CnnRender.lean:1488`) uses:

    convBack  convWeightSgd  convBiasSgd
    bnPerChannelF  bnPerChannelBack  bnSig  bnGammaSgd  bnBetaSgd
    dense  denseF  dotOut

Its signature has **no** `bf16`/`fp8` flags. Compare `cifar8AdamTrainStepFaithfulB`
(`:1247`), the batched peer, which ends:

    (replicas : Nat := 1) (opt : CifarOpt := .adamw) (bf16 : Bool := false) (fp8 : Bool := false)

| op | bf16 twin? |
|---|---|
| conv fwd / back / weight-grad | ✅ batched only (`convBf16`, `convBackBatchedBf16`, `convWeightGradBBf16`) |
| dense / dot | ✅ batched (`denseRowBf16`, `dotInBf16`, `rowDenseWeightGradBBf16`) |
| **`bnPerChannelF` / `bnPerChannelBack`** | ⛔ **none, either family** |

⭐ So exactly **one op pair** is missing, and it is BatchNorm's.

---

## 2. Two routes; take (b)

**(a) Add bf16 to the per-example BN emitter.** ⛔ Needs per-example bf16 twins for conv as well
— none exist. Two problems, not one.

**(b) Write `cifar8BnTrainStepFaithfulB` — BN in the batched family.** ⭐ The conv and dense bf16
twins are already there; only BN needs new ones. It also matches where the book is going: every
ImageNet chapter is batched.

---

## 3. The work, in order

1. **`bnPerChannelFBf16` / `bnPerChannelBackBf16`** as `SHlo` constructors.
   ▶ Model them on `convBackBatchedBf16` (`StableHLO.lean:1019`, `den` case at `:2234`). Its
   comment states the discipline: *"dgrad is itself a convolution, so it takes the same emit shape
   and the same `den` discipline as the forward."*
   ⚠ BN's backward is the **three-term** one Chapter 4 proves. The rounding has to thread through
   all three terms, which is where this stops being mechanical. Budget the session here.
   ⚠ See [[bf16-conv-emit-shape]]: a bf16 op needs a **bf16-TYPED result plus a convert**. The
   f32-result shape works for `dot` and is SILENTLY FOLDED AWAY on conv.

2. **`cifar8BnTrainStepFaithfulB`** — port the eight-conv BN train step to the batched family,
   with `(bf16 : Bool := false)` trailing and defaulted so existing positional calls are unchanged.

3. **Render** `verified_mlir/cifar8w_bn_bf16_{sgd,mom,adam}_train_step.mlir`.
   ⚠ A new `verified_mlir/` file must be added to `proofs.yml`'s diff list or
   `check_render_coverage.py` — the only thing that catches it — stays silent. See
   [[render-guard-on-new-artifact]].

4. **`cifar8w-bn-bf16-ablation`** binary, `apps/ablation/`, three arms, mirroring
   `MainCifar8WideBnAblation.lean`. ⚠ Constant LR: `trainAdamSched cfg d LR 0.9 0.999 0 "v" 1.0 1.0`
   (`expDecayRate = 1.0` ⇒ exactly flat; `warmupEpochs = 0` ⇒ warmup never fires).

5. **Run n=5** against the fp32 BN baseline already in
   `runs/2026-09-01-cifar8w-6arm-constlr/bn_s{1..5}.log` (medians: SGD 74.50, mom 76.35,
   AdamW 74.29; **zero NaN in all 15 arm-runs**, which is what makes it a clean baseline).
   ~12 min/run, 5 runs, six GPUs ⇒ under half an hour.

6. **Rewrite §4.5's Lever 3** around the result. It is currently 324 words ending in a forward
   reference to this work; replace that with the measurement.

---

## 4. What "success" is

A table of fp32 vs bf16 on the **BN** net, five seeds, same op family, constant LR. If the medians
agree inside the run-to-run spread and bf16 adds no NaN, Chapter 5's use of bf16 is licensed by
measurement rather than assumption.

⚠ **A null result is still a result.** If bf16 costs the normalized net a point, that belongs in
the book and Chapter 5's recipe section needs to say so.

---

## 5. ▶ TODO, deliberately deferred: fp8

`cifar8w-fp8-ablation` exists and runs, but its render is honest about itself
(`CnnRender.lean:2025`): *"forward convs only for now (`convBackBatchedF8` / `convWeightGradBF8` do
not exist yet), and UNSCALED, so this is a lowering probe rather than a trainable arm: E4M3 maxes
at 448 and XLA synthesises scale = 1.0 when given no scale operand."*

So an fp8 arm needs the f8 backward ops **and** a scaling story before it means anything. Chapter 4
now says this in one sentence and drops the column. Nothing in the book depends on fp8 today.

---

## 6. State of the tree as this was written

* Chapter 4 is converted to constant LR: `apps/ablation/MainCifar8Wide{Bn,,Batched,Bf16,Fp8}Ablation.lean`
  all pass `0 "variant" 1.0 1.0`, and `trainAdamSched` / `trainAdamSchedE4M3` grew a `constant lr`
  log spelling. ⚠ The `cosine` and `exp` spellings are byte-identical to before, because every
  Imagenette and ImageNet transcript quotes them.
* 25 runs in `runs/2026-09-01-cifar8w-6arm-constlr/` — bn, nobn, bf16, fp8, fp32b at n=5.
* ⚠ Uncommitted at time of writing.

---

## 7. Finishing Chapter 4, which is the point

Chapter 4 is complete except for Lever 3, which currently ends by handing the measurement to this
work. Two outcomes, both of which finish the chapter:

**If the twin lands.** Replace §4.5's Lever-3 closing paragraphs with the fp32-vs-bf16 table on the
BN net (five seeds, same op family, constant LR) and state the licence plainly: Chapter 5 trains in
bf16, and here is the measurement that says that is free. ⭐ Keep the op-family finding — it is
independently true and it is the only place in the book that warns about it.

**If it does not.** Cut Lever 3's forward reference to a single sentence — "a bf16 BatchNorm render
does not exist yet, so the normalized case is unmeasured" — and let Chapter 5 introduce precision
from scratch. ⚠ Do NOT leave the chapter promising a measurement that no longer has a session
behind it; that is how §4.5 acquired the divergence claim it just lost.

Either way the chapter is then done and can be committed and pushed, which is what unblocks the
ResNet session on the other machine (`planning/resnet_chapter_pass.md`).

### 7.1 What is left in Chapter 4 regardless

* Nothing. §4.1's transcript, both spec listings, all three levers and the prose are converted to
  constant LR and verified against `runs/2026-09-01-*`. Lever 3 is the only section whose *claim*
  is provisional.
* ⚠ Re-read §4.5 once end-to-end before committing. Two claims in this session's drafts were
  contradicted by numbers printed two paragraphs above them, and both were caught by reading the
  rendered PDF rather than the source.

