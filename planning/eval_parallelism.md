# Per-epoch eval runs on ONE of four GPUs — scoped, measured, not built

**Written 2026-09-17** during the ConvNeXt-T verified 300-epoch run
(`runs/2026-09-17-cnx-verified-300ep/`). ⚠ Nothing here is implemented. This is the measurement and
the decision, so the next person does not re-derive either.

Brett, 2026-09-17: *"we talked about the eval thing before but never did the 4x gpu testing"* — so
this file exists to stop it being re-discussed a third time without a number attached.

---

## 1. The finding, measured on a live run

Sampling `nvidia-smi` across an epoch boundary of the ConvNeXt run:

    08:30:28  step 5000   util: 91 %  0 %  0 %  0 %
       ...       (97 s of this)
    08:32:00  step 5000   util: 97 %  0 %  0 %  0 %
    08:32:05  marker -> e13   util: 69 % 53 % 100 % 90 %

**Eval takes ~100 s per epoch with GPU 0 at ~90% and GPUs 1–3 at 0%.**

⭐ **It is fleet-wide, not a ConvNeXt quirk.** Every eval forward in `verified_mlir/` is
single-replica — `grep -c all_reduce` is **0 on every `*_fwd.mlir` and every `*_fwd_eval.mlir`**.
The train steps are 4-replica; the eval forwards never were.

⚠ **And it is COMPUTE-bound on that one device, not launch-bound.** GPU 0 sits at ~90% for the
whole window, so a wider eval batch (fewer, bigger invokes on the same card) buys almost nothing.
The only real lever is using the other three cards.

The checkpoint is NOT the cost and this was checked: the 343 MB `.bin` and its `.epoch` marker are
written 100 ms apart, so essentially all of the ~105 s boundary overhead is eval.

## 2. What it costs

| net / run | epochs | measured per-epoch eval |
|---|---|---|
| ConvNeXt-T 300ep (this run) | 300 | **~100 s** (direct, with GPU util) |
| ResNet-34 bf16 90ep | 90 | **~70 s** (886 s mean epoch − 816 s of steps at the probe's 163 ms) |

Both score the same 50,000-image split, so 70–100 s is the band. Over the six-net sweep
(R34 90 + R50 90 + MNv2 350 + ENet 350 + ConvNeXt 300 + ViT 300 = **1,480 epochs**) that is
**29–41 h of wall clock with three of four GPUs idle**, of which ~75% is recoverable.

For this run alone: 300 × 100 s = **8.3 h of 102 h (8.2%)**.

## 3. The three options, costed

| option | saving (a full 300ep run) | keeps the per-epoch curve? |
|---|---|---|
| eval every 5 epochs | 6.7 h | partly — 60 points instead of 300 |
| eval every 10 epochs | 7.5 h | weakly — 30 points |
| **4-replica eval** | **~5.9 h** | **yes, all 300** |

Arithmetic for the DP row, off the measured 1,209 s/epoch mean: ~1,104 s of steps + ~100 s eval
+ ~5 s checkpoint. At 4 replicas eval goes ~100 -> ~30 s (not a clean 4x — host-side slicing, the
fixed per-invoke cost and the tail do not shard), so per-epoch **1,209 -> ~1,139 s** and a 300-epoch
run **100.8 h -> ~94.9 h**.

⚠ **5 → 10 buys only another 0.8 h.** Almost the whole win is in the first step, so there is no
reason to go past every-5.
⭐ **DP eval gets ~80% of the every-10 saving while losing NO evidence**, because the problem was
never that we evaluate too often — it is that we evaluate on 25% of the box.

## 4. ⭐ THE DECISION (Brett, 2026-09-17)

* **v1 / first-of-its-kind runs: eval EVERY epoch. Do not change this.** *"for a v1 better to be
  watching too closely."* The per-epoch curve IS the deliverable on a pair run: the 50-epoch window
  table and the **epochs-above-reference count** are the decisive outputs, and EfficientNet's tell
  was *"0 of 300"* — at every-10 that becomes "0 of 30", a 10× weaker statement against a reference
  that has all 300 points.
* **Followups / re-runs of a known shape: every 5 is fine.** *"every 5 gives you a decent curve at
  this scale."* 60 points over 300 epochs still gives 12 per 50-epoch window.
* ⛔ **Not applied to the in-flight ConvNeXt run** — it would cost a restart, and this run is a v1.

## 5. What building either one actually takes

### A. `LEAN_MLIR_EVAL_EVERY=N` — the cheap one

`LEAN_MLIR_SKIP_EVAL` today is **all-or-nothing** (`.isSome`, `VerifiedTrain.lean:1534`), so the
interval does not exist and is a small driver change: an interval knob, plus *always* evaluating
the FINAL epoch regardless (a run whose last epoch is unscored has no headline number), plus the
banner saying which it did — the `SKIP_EVAL` precedent is that a silently-unscored run prints a
perfectly confident 0.000000% (`wilson95`'s docstring, line ~200).

### B. 4-replica eval — the one worth doing

⭐⭐ **RE-SCOPED 2026-09-18 after reading the FFI: NO RENDER CHANGE IS NEEDED and no committed
artifact moves.** A forward has no cross-replica op, so the *same module* computes the same thing
at any replica count — data-parallel inference is just "run the same program on 4 devices with 4
different input slices". The only thing pinning it to one device is a heuristic in
`ffi/pjrt_ffi.c:735`:

    int reps = (g_replicas > 1 && strstr(mlir, "all_reduce")) ? g_replicas : 1;

with the comment right above it explaining why — *"A module with no cross-replica op computes the
same thing at any replica count and is only ever invoked single-device (the eval forward is exactly
this), so compile it for one replica — otherwise Execute rejects it with 'Attempted to execute with
1 argument lists when local device count is 2'."* That comment is correct about the CONSTRAINT and
wrong about the CONCLUSION: the fix is to pass 4 argument lists, not to compile for one device.

So the work is three pieces and none of them is a renderer:

1. **`ffi/pjrt_ffi.c`** — let a session be told its replica count instead of inferring it from
   `all_reduce`, and give the forward path a replicated invoke returning **per-replica** outputs.
   ⭐ `pjrt_ffi_invoke_f32_dp` and the replica-major buffer layout (`buf[replicas * n]`,
   `sess->replicas`) ALREADY EXIST for the train step, and sessions already carry independent
   replica counts — which is exactly why a 1-replica eval session coexists with a 4-replica train
   session today. This is adaptation, not new machinery. **The uncertain piece; timebox it.**
2. **`LeanMlir/VerifiedTrain.lean`** — the eval loop feeds `4 x evalBs` rows per iteration and
   scores `min (4*evalBs) (nEval - bi*4*evalBs)` real rows. ⭐ The ragged tail is ALREADY handled
   correctly and generically (`F32.sliceImagesPad` + the `min` guard at line 2480), and `evalBs` is
   already READ OFF THE ARTIFACT (`fwdRenderedShape`), so nothing is hardcoded. ~20-30 lines.
3. **The gate** — below. Non-negotiable.

⛔ Two cheaper-looking options examined and REJECTED:
* **Widen the eval render's batch** (196 invokes instead of 782 on the one card). `evalBs` being
  read off the artifact makes this look free, but `convnextin_fwd` is byte-prefix-paired with the
  train steps by `check_fwd_prefix`/`check_adam_prefix`, so it must keep the train batch. A
  separate wide eval artifact means new files plus audit updates, for a launch-overhead win only.
* **A bf16 eval render.** The eval forward is f32 and the card is running at roughly 20% of its
  fp32 peak (225 TFLOPs of work in ~100 s), so there is headroom — but eval in bf16 moves logits
  and can flip borderline predictions, so it needs its own numeric gate. Not free, and orthogonal.

⛔⛔ **AND IT NEEDS ITS OWN GATE BEFORE IT IS TRUSTED ANYWHERE.** An eval that silently scores a
different subset — drops the ragged tail, double-counts a shard, or scores 4 copies of one shard —
is *exactly* this repo's recurring bug class, and it is the one place a defect reads as an accuracy
change rather than a crash. The precedents are not hypothetical: `argmax10` confined every ImageNet
prediction to labels 0..9 for months, and the C4 denominator fix moved every number. The gate is
cheap and obvious: **score one fixed checkpoint through the 1-replica and the 4-replica paths and
require the correct-count to be EQUAL, not close** — same weights, same images, only the sharding
moves. Plus a control that deliberately mis-shards and must FAIL.
⚠ `n=50,000` is not divisible by `4 × 64 = 256` (it is 195 batches + a 80-image tail), so the
ragged tail is the first thing to get wrong and the first thing the gate must cover.

## 6. Next — FOLDED INTO THE cnxInit REWORK (Brett, 2026-09-18)

The e65 finding that the two arms do not share a weight init (`runs/2026-09-17-cnx-verified-300ep/`
§7.0) means ConvNeXt is being re-run anyway, so **B ships with that re-run** rather than waiting.

| piece | effort | risk |
|---|---|---|
| verified-side `cnxInit` (~15 lines, 2 files, host-side) | ~1.5 h | low — `vitInit` is the template |
| 4-replica eval (C + driver + gate) | 3–6 h | **medium — the C invoke is the unknown** |
| combined re-run | ~95 h (vs ~101 h without B) | — |

⭐ **B pays for itself inside this single run** — ~6 h of work against ~6 h of saved wall clock —
and every run after it banks the saving for free. Across the six-net sweep's 1,480 epochs it is
worth 22–31 h.

⚠ **Timebox piece 2.** If the replicated forward invoke is not working after ~4 h, launch with
`cnxInit` alone and keep B as its own follow-up; the re-run must not be blocked on the harder half.
A is then the cheap fallback, and Brett's rule stands: **v1 every epoch, followups every 5.**
