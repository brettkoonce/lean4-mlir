import LeanMlir.VerifiedNets

/-! # `resnet34-ablation` — §5.6's leave-one-out recipe ablation, one arm per invocation

    `.lake/build/bin/resnet34-ablation data <arm> [bf16]`

ResNet-34 / Imagenette, 80 epochs, batch 32, on the VERIFIED path (XLA/PJRT). Every arm keeps
the full recipe except the one component it names, so its delta measures that component's
contribution *given everything else is present* — not the order-dependent story an additive
ladder gives.

**One arm per invocation, deliberately.** The Chapter 4 ablation binaries run their arms in
sequence, which is right at 40 epochs on CIFAR; here an arm is ~80 minutes, so eight of them in
series is a working day and in parallel across four cards it is under three hours.

⚠ **Three arms are RENDERS, not flags, and the reason is where the constant lives.** Warmup, the
schedule and augmentation are host-side, so they are arguments here. Weight decay and label
smoothing are baked into the train step (`optConstsB`'s `%wd`; the α/K pair in the smoothed-CE
cotangent), so `nowd` and `nols` load their own artifacts. The optimizer arms likewise select a
different render. Every one of those artifacts differs from `resnet34_adam` in constants only.

⚠ `LEAN_MLIR_CKPT_TAG` is set per arm by the driver script, not here — without it the arms share
a checkpoint lineage and the second one resumes the first at epoch 80 and reports `done`.
-/

def cfg : VerifiedConfig where
  epochs    := 80
  batchSize := 32

/-- The precision suffix, appended to every variant. ⚠ It TRAILS the ablation marks (`adamwd00bf16`,
    not `adambf16wd00`), which is `r34AdamVariant`'s spelling and the `#guard`s pin it — a suffix
    assembled in the wrong order here would name an artifact that does not exist. -/
private def pfx (bf16 : Bool) : String := if bf16 then "bf16" else ""

def main (argv : List String) : IO Unit := do
  let d    := argv.head?.getD "data"
  let arm  := (argv.drop 1).head?.getD "full"
  let bf16 := (argv.drop 2).head?.getD "" == "bf16"
  let p    := pfx bf16
  -- Full recipe: AdamW, cosine + 3-epoch warmup, wd and label smoothing baked, augmentation on.
  let full : String → IO Unit := fun d =>
    resnet34Verified.toNet.trainAdamSched cfg d 0.001 0.9 0.999 3 s!"adam{p}"
  IO.println s!"════════ resnet34 ablation — arm `{arm}` ({if bf16 then "bf16" else "fp32"}) ════════"
  match arm with
  | "full"    => full d
  -- host-side arms: same artifact, different arguments
  | "nowarm"  => resnet34Verified.toNet.trainAdamSched cfg d 0.001 0.9 0.999 0 s!"adam{p}"
  | "nocos"   => resnet34Verified.toNet.trainAdamSched cfg d 0.001 0.9 0.999 3 s!"adam{p}" 1.0 1.0
  -- ⚠ `noaug` is `full` under LEAN_MLIR_NO_AUG=1, set by the caller. It is listed here so the
  -- arm names form one closed set and a typo is an error rather than a silent `full`.
  | "noaug"   => full d
  -- render arms
  | "nowd"    => resnet34Verified.toNet.trainAdamSched cfg d 0.001 0.9 0.999 3 s!"adamwd00{p}"
  | "nols"    => resnet34Verified.toNet.trainAdamSched cfg d 0.001 0.9 0.999 3 s!"adamls0{p}"
  -- Optimizer arms, both on the heavy-ball render (`resnet34_mom`).
  -- ⚠⚠ `noadam` IS NOT "vanilla SGD". `R34Opt` has no plain-SGD case, so the leave-one-out arm
  -- for the optimizer swaps AdamW for Nesterov momentum with the rest of the recipe intact. The
  -- pre-2026-04 table called this row "vanilla SGD, lr 0.01" and it was never that; naming it for
  -- the render it loads is the only spelling that cannot drift from what ran.
  | "noadam"  => resnet34Verified.toNet.trainAdamSched cfg d 0.01 0.9 0.999 3 s!"mom{p}"
  -- ⚠ `bare` is that SAME render with warmup and the schedule off too, so it differs from
  -- `noadam` by the schedule alone. Read it as "what the modern recipe buys in total", never as
  -- a leave-one-out row.
  | "bare"    => resnet34Verified.toNet.trainAdamSched cfg d 0.01 0.9 0.999 0 s!"mom{p}" 1.0 1.0
  -- ⭐ `bare02` — Chapter 4's momentum settings, transplanted VERBATIM: μ 0.9, lr **0.02**,
  -- no warmup, constant rate (`MainCifar8WideBnAblation.lean`'s middle arm, argument for
  -- argument). It differs from `bare` in the learning rate ALONE, which is what makes the pair a
  -- measurement of the rate rather than of the recipe — and what says whether `bare`'s 84.43%
  -- is the optimizer's ceiling here or just an under-stepped one. A momentum rate that suits a
  -- 9-layer CIFAR net at 32² need not suit a 34-layer ResNet at 224².
  | "bare02"  => resnet34Verified.toNet.trainAdamSched cfg d 0.02 0.9 0.999 0 s!"mom{p}" 1.0 1.0
  -- ⭐ `baresgd10` — the bare configuration under PLAIN SGD at the step-matched rate. `bare` is
  -- momentum at 0.01, i.e. an effective 0.1; this is 0.1 with no velocity. If the bare row's
  -- advantage over `sgd` is also just step size, the two land together and momentum contributes
  -- nothing here either — the same test `sgd10` runs against `noadam`, one configuration down,
  -- and now with no schedule or warmup left to absorb the difference.
  | "baresgd10" => resnet34Verified.toNet.trainAdamSched cfg d 0.1 0.9 0.999 0 s!"sgd{p}" 1.0 1.0
  -- ⭐⭐ THE OPTIMIZER LADDER'S BOTTOM RUNG — plain SGD, full recipe otherwise, so `full` /
  -- `noadam` / `sgd` differ ONLY in the update rule. `noadam` alone cannot answer what the
  -- optimizer buys, because momentum is most of it.
  -- ⚠⚠ TWO RATES, AND THE PAIR IS THE POINT. Heavy-ball at μ = 0.9 amplifies the effective step by
  -- 1/(1−μ) = 10×, so plain SGD at the SAME nominal 0.01 is not "momentum removed" — it is
  -- momentum removed AND the step shrunk tenfold, which is the same conflation `noadam` makes one
  -- level up. `sgd` holds the nominal rate; `sgd10` holds the EFFECTIVE one at lr 0.1. Whatever
  -- separates them is step size, and whatever survives both is momentum.
  | "sgd"     => resnet34Verified.toNet.trainAdamSched cfg d 0.01 0.9 0.999 3 s!"sgd{p}"
  | "sgd10"   => resnet34Verified.toNet.trainAdamSched cfg d 0.1  0.9 0.999 3 s!"sgd{p}"
  -- ⭐⭐ THE MOMENTUM-BASE LADDER (`m*`). The same leave-one-out set rooted at heavy-ball
  -- momentum, lr 0.01, instead of AdamW. AdamW's per-parameter normalisation absorbs exactly the
  -- work warmup, the schedule and the decay are doing, so an ablation rooted there may understate
  -- all of them at once; rooted at momentum nothing is absorbing it. The comparison of the two
  -- ladders is the measurement, not either one alone.
  -- ⚠ `mfull` IS the `noadam` arm and `mbare` IS `bare` — same arguments, same artifact. They are
  -- named here so the momentum ladder can be requested as one set, and they must NOT be re-run
  -- into a second log: use the existing numbers.
  | "mnowarm" => resnet34Verified.toNet.trainAdamSched cfg d 0.01 0.9 0.999 0 s!"mom{p}"
  | "mnocos"  => resnet34Verified.toNet.trainAdamSched cfg d 0.01 0.9 0.999 3 s!"mom{p}" 1.0 1.0
  | "mnoaug"  => resnet34Verified.toNet.trainAdamSched cfg d 0.01 0.9 0.999 3 s!"mom{p}"
  | "mnowd"   => resnet34Verified.toNet.trainAdamSched cfg d 0.01 0.9 0.999 3 s!"momwd00{p}"
  | "mnols"   => resnet34Verified.toNet.trainAdamSched cfg d 0.01 0.9 0.999 3 s!"momls0{p}"
  | other     =>
    IO.eprintln s!"⛔ unknown arm `{other}`. One of: full nowarm nocos noaug nowd nols noadam bare bare02 sgd sgd10 mnowarm mnocos mnoaug mnowd mnols baresgd10"
    IO.Process.exit 1
