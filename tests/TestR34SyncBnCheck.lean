import LeanMlir.SyncBnCheck
import LeanMlir.VerifiedNetsCore
import LeanMlir.Proofs.Codegen.ResNet34RenderB

/-! # `resnet34-syncbn-check` — synchronised BatchNorm: 2×b IS 1×2b

    lake build resnet34-syncbn-check
    unset CUDA_VISIBLE_DEVICES
    PJRT_REPLICAS=2 .lake/build/bin/resnet34-syncbn-check

The identity every `*-dp-check` and `shard-check` could NOT state for a batch-BN net. Until
2026-09-21 a data-parallel R34 render normalised each replica over its own `b` rows, so
`DP([xA|xB])` was `mean(single_b(xA), single_b(xB))` — a step on the mean of two per-replica
losses, and provably not the single-device step at `2b` (`dpMeanGrad_ne_globalBatchGrad`;
`lakefile.lean`'s own words: "R34 is about SPLITTING a batch — 2×32 really is not 1×64").

With the sync-BN render (`planning/global_bn_verified.md` §2b: every BN layer all-reduces its
μ, then its Chan-corrected σ² (`σ²_r + (μ_r − μ)²`), before normalising; its backward all-reduces
the two dy-reductions; and the γ gradient reads the same global `x̂`) the replicas compute their
shards of ONE global-batch function, and the identity becomes exact up to float reduction order:

    TEST     DP_sync( [xA | xB] )  ==  single_2b( [xA | xB] )
    CONTROL  DP_sync( [xA | xB] )  !=  mean( single_b(xA), single_b(xB) )

The CONTROL is the old identity, and it must now FAIL by a margin: if it still held, the
statistics would not be synchronised and the TEST would be passing for the wrong reason (both
sides per-replica). `dpSyncGrad_eq_globalBatchGrad` / `den_bnSyncF_allReduce` … are the
ℝ-level statements this gates the emitted bytes against — the first numeric check any of the
seven sync ops' MLIR has had.

**What is compared, and why all of it.** With `m = 0` fed in, `m' = 0.1·g` is linear in the
gradient (the `shard-check` trick). Here the gradient itself is equal, not just averagable — the
per-replica divisor `1/b` and the collective's `1/2` compose to the single device's `1/(2b)`
(`DataParallelSync.lean`, "the 1/R") — so `v' = 0.001·g² + 0.999·v` and Adam's `θ'` are equal too,
and so are the 72 handed-back BN statistics (global μ / σ² on every replica, versus the 2b batch's
own). Every output region is checked; only the report-only `%loss` slot is skipped, since the DP
render logs replica 0's shard loss and does not all-reduce it.

**How the columns are read.** Two more runs bracket the comparison. The one-replica SYNC graph
on the whole batch (every collective empty) splits TEST into COLLECTIVE (DP vs it — the
collective composing the sync ops) and FORMULATION (it vs the two-pass graph — the arithmetic of
the exchange). And a SENSITIVITY probe — the two-pass graph on the same batch perturbed by
1e-4·N(0,1) per pixel — is the yardstick for the gradient columns: at this random-init operating
point it moves the two-pass graph's OWN `m'` by ~0.2, so the gradient can only ever be compared to
~1e-3 of that, while the statistics are compared tightly.

⛔ **History (2026-09-21).** The first render exchanged `[μ ‖ E[x²]]` in one round and every
consumer formed `σ² = E[x²] − μ²`. This gate measured it 2e-4 off in the statistics after 36
layers and 15 % off in `m'`, with the sensitivity probe at 0.22 — i.e. the ops were right and the
f32 arithmetic was not (`ε·E[x²]/σ²` per layer, compounding). Chan's two-round exchange replaced
it; the numbers below are its.

Needs TWO GPUs and the XLA backend (collectives do not exist on the IREE path). The runner is
`LeanMlir.SyncBnCheck`, shared with the MobileNetV2, EfficientNet-B0 and ImageNet gates; the
committed artifacts are `resnet34_adam_train_step` (1×32) and `resnet34_adamdp_train_step`
(2×32, sync) — or the DP step named by the first argument — and the 1×64 two-pass step
(`resnet34_adam64_train_step`'s renderer) and both one-replica sync graphs are rendered to
`.lake/build/` at run time. ~30 s.
-/

def main (args : List String) : IO Unit :=
  SyncBnCheck.run
    { slug := "resnet34", net := resnet34Verified.toNet, bs := 32
      sgPath := "verified_mlir/resnet34_adam_train_step.mlir"
      dpPath := args[0]?.getD "verified_mlir/resnet34_adamdp_train_step.mlir"
      render := fun B fs => Proofs.StableHLO.resnet34AdamTrainStepFaithfulB B 10 "1.0e-05"
        (forceSync := fs)
      entry := fun B r => s!"m.resnet34_{Proofs.StableHLO.r34AdamVariant B r}_train_step" }
