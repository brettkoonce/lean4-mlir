import LeanMlir.SyncBnCheck
import LeanMlir.Proofs.Codegen.ResNet34RenderB
import LeanMlir.Proofs.Codegen.MobileNetV2RenderB
import LeanMlir.Proofs.Codegen.EfficientNetRender
import LeanMlir.Proofs.Codegen.ResNet50RenderB
import LeanMlir.Proofs.Codegen.MobileNetV4RenderB

/-! # `imagenet-syncbn-check` — synchronised BatchNorm at the ImageNet shape: 4×64 IS 1×256

    lake build imagenet-syncbn-check
    unset HIP_VISIBLE_DEVICES
    PJRT_REPLICAS=4 .lake/build/bin/imagenet-syncbn-check resnet34        # momdp64bf16
    PJRT_REPLICAS=4 .lake/build/bin/imagenet-syncbn-check mobilenetv2     # rmsdp64bf16
    PJRT_REPLICAS=4 .lake/build/bin/imagenet-syncbn-check efficientnet    # rmsdp64bf16
    PJRT_REPLICAS=4 .lake/build/bin/imagenet-syncbn-check resnet50        # momdp, 4×32 vs 1×128
    PJRT_REPLICAS=4 .lake/build/bin/imagenet-syncbn-check resnet50bce     # lambdp bce @160, 4×32
    PJRT_REPLICAS=4 .lake/build/bin/imagenet-syncbn-check mnv4            # adamdp64bf16
    LEAN_MLIR_MEM_FRACTION=0.97 PJRT_REPLICAS=4 … <net> f32               # the f32 peer

⚠ The raised arena is for f32 only: in bf16 it crowded out the host-transfer staging on
`resnet50bce` (`CUDA_ERROR_OUT_OF_MEMORY` on a d2h, not a compile) where the default runs clean.

**ResNet-50 is gated at 4×32 against 1×128**, not at its committed 4×64: XLA's peak for the 1×256
reference is 14.14 / 14.35 GiB (bf16 / f32), 94–95 % of even the raised arena. Its DP step is
rendered at run time (`dpPath := ""`) by the same `resnet50TrainStepFaithfulB` that writes the
committed `momdp64` / `lambdp64bce` bytes, and `regen_verified_mlir.sh check` pins those to it.
`resnet50bce` is the LAMB + BCE-with-logits loss at 160² that the A3 renders take; the
accumulating (`acc`) renders add an accumulator after the collective, which the runner cannot feed
and the sync-BN identity does not involve.

The Imagenette gates (`resnet34-syncbn-check`, `mobilenetv2-syncbn-check`,
`efficientnet-syncbn-check`) run f32 at 2×32. The artifacts the ImageNet pairs train from are
bf16, 4-replica and 224²/1000-class, and this gate runs `LeanMlir.SyncBnCheck`'s columns on
those: the committed 4×64 DP step against a 1×256 single-device step rendered at run time. XLA's
compiled peak at 256 is 6.2 / 5.8 / 6.7 GiB for R34 / MNv2 / B0 in bf16, inside the default
arena; R34's f32 step at 256 is not, hence the raised fraction on the f32 line.

The DP artifact per net is the one its book pair trained from: R34 `momdp64bf16`, MNv2
`rmsdp64bf16`. B0's pair trained from `emarmsdp64dropdobf16`, which adds EMA and host-fed
drop-path / dropout masks; `rmsdp64bf16` is the same net, optimizer and precision without them,
the scope B0's DP twin is stated at.

**The shards are shifted** (`shardShift := 0.5`): at 224² the statistics of 64 iid images sit
within ~1e-3 of the global ones, so unshifted shards gave R34 a CONTROL of 2.7e-3 against a TEST
of 1.0e-3, and Chan's `(μ_r − μ)²` term was too small to see. Shifted, CONTROL is 0.72 / 0.074 /
0.40 and the first BN layer is split-exact in both precisions.

**bf16 bounds, measured 2026-09-21.** The sharp criteria are precision-independent and hold at
1e-5 in bf16: the first BN layer's split error and the DUPLICATED statistics are 0 on all three
nets. What bf16 moves is every comparison between two compiled programs. FORMULATION shows the
floor with no collective in either graph: one function, one shape, gradient columns 1.3e-2 /
1.5e-2 / 7–9e-2 in bf16 against 2.2e-4 / 6.4e-4 / 1.4e-3 in f32. REPEAT (one program, run twice)
is 0, so the floor is between programs, not between runs. That is consistent with an f32
last-bit difference between two programs becoming a bf16-ulp difference at the next f32→bf16
convert, which the random-init backward then amplifies. Per net, bf16 vs f32:

    net    stats TEST               FORMULATION stats   DUPLICATED gradient norm-rel
    R34    0.70–1.0e-3 / 1.3e-4     0 / 0               1.27e-2      / 7.0–7.2e-4
    MNv2   4.3–5.2e-3  / 3.6–4.8e-4 0 / 0               1.62–1.70e-2 / 1.0e-3
    B0     2.5–3.1e-3  / 1.6–2.0e-4 1.4–2.0e-3 / 0      1.19–1.20e-2 / 6.8–7.0e-4
    R50    6.7–8.6e-3  / 1.7–2.1e-3 0 / 0               2.05–2.07e-2 / 1.0e-3
    R50bce 1.08–1.22e-2 / 2.2e-3    0 / 0               1.69–1.72e-2 / 1.0e-3
    MNv4   6.9e-3–1.16e-2 / 1.2–1.3e-3  0 / 0           1.83–1.85e-2 / 1.2e-3

⚠ The two deeper nets (53 and 77 BN layers) split at the size of their own forward's
conditioning: the statistics SENSITIVITY column (the two-pass graph against itself on `x`
perturbed by 1e-4) reads 1.4–1.6e-2 bf16 and 1.3–1.7e-3 f32 on R50 and MNv4, and TEST sits at or
under it in every run. Their bounds are ≈ 2× that; the exchange itself is carried by the sharp
columns, which are 0 on both.

A sum-not-mean collective makes every all-reduced gradient `4g`, a relative error of 3 — 60×
over the bf16 bound.

Four GPUs, XLA backend.
-/

open Proofs.StableHLO in
/-- The gate for one net at 4×64, bf16 unless `f32`. -/
def imagenetCfg (net : String) (bf16 : Bool) : Option SyncBnCheck.Cfg :=
  let p := if bf16 then "bf16" else ""
  -- bf16's DUPLICATED floor is two programs' rounding: gradient norm-rel 1.19–1.70e-2, three nets
  let dupTol : Float := if bf16 then 5e-2 else 5e-3
  match net with
  | "resnet34" => some
    { slug := s!"resnet34in{p}", net := resnet34ImagenetVerified.toNet, bs := 64, replicas := 4
      sgPath := ""
      dpPath := s!"verified_mlir/resnet34in_momdp64{p}_train_step.mlir"
      render := fun B fs => resnet34AdamTrainStepFaithfulB B 1000 "1.0e-05" 1 .heavyBall
        "resnet34in" false bf16 (forceSync := fs)
      entry := fun B r =>
        s!"m.resnet34in_{r34AdamVariant B r .heavyBall false false false "" bf16}_train_step"
      -- heavy-ball keeps its velocity in `v` and passes `m` through
      gradSlot := 2, gradLabel := "v' = g + wd·θ", vZero := true, shardShift := 0.5
      statsTol := if bf16 then 3e-3 else 1e-3, dupTol }
  | "mobilenetv2" => some
    { slug := s!"mobilenetv2in{p}", net := mobilenetv2ImagenetVerified.toNet, bs := 64
      replicas := 4, sgPath := ""
      dpPath := s!"verified_mlir/mobilenetv2in_rmsdp64{p}_train_step.mlir"
      render := fun B fs => mobilenetv2AdamTrainStepFaithfulB B 1000 "1.0e-5" 1 false
        "mobilenetv2in" .rmsprop bf16 (forceSync := fs)
      entry := fun B r => s!"m.mobilenetv2in_{mnv2AdamVariant B r .rmsprop bf16}_train_step"
      gradLabel := "m' = g/√(s'+ε)", shardShift := 0.5
      -- the Imagenette gate's 3e-3 (depth-compounded rounding, 52 layers) holds in f32; bf16
      -- compounds from the second layer on (the depthwise twins)
      statsTol := if bf16 then 1e-2 else 3e-3, dupTol }
  | "efficientnet" => some
    { slug := s!"efficientnetin{p}", net := efficientnetImagenetVerified.toNet, bs := 64
      replicas := 4, sgPath := ""
      dpPath := s!"verified_mlir/efficientnetin_rmsdp64{p}_train_step.mlir"
      -- B0's loss divisor is a string: each run-time render passes its own batch
      render := fun B fs => efficientnetAdamTrainStepFaithful B 1000 "1.0e-5" "0.100000" ""
        s!"{B}.0" 1 false "efficientnetin" .rmsprop (bf16 := bf16) (forceSync := fs)
      entry := fun B r =>
        s!"m.efficientnetin_{enetAdamVariant B r .rmsprop false false false bf16}_train_step"
      gradLabel := "m' = g/√(s'+ε)", shardShift := 0.5
      -- B0 is the one net whose sync and two-pass graphs round apart in bf16 at the same batch
      statsTol := if bf16 then 6e-3 else 1e-3, formTol := if bf16 then 5e-3 else 1e-5, dupTol }
  | "resnet50" => some
    { slug := s!"resnet50in{p}", net := resnet50ImagenetVerified.toNet, bs := 32, replicas := 4
      sgPath := "", dpPath := ""
      render := fun B fs => resnet50TrainStepFaithfulB B 1000 "1.0e-05" 1 .heavyBall
        "resnet50in" (bf16 := bf16) (forceSync := fs)
      renderDp := fun B R => resnet50TrainStepFaithfulB B 1000 "1.0e-05" R .heavyBall
        "resnet50in" (bf16 := bf16)
      entry := fun B r =>
        s!"m.resnet50in_{r34AdamVariant B r .heavyBall false false false "" bf16}_train_step"
      gradSlot := 2, gradLabel := "v' = g + wd·θ", vZero := true, shardShift := 0.5
      -- 53 layers: the split error is the forward's own conditioning at random init, not the
      -- exchange — statistics TEST 6.7e-3 / 8.6e-3 bf16 and 1.7e-3 / 2.1e-3 f32 on two runs each,
      -- against a SENSITIVITY-to-1e-4 of 1.4e-2 / 1.4e-3, variance error growing smoothly to
      -- 1.6e-2 / 6.3e-3 by the last layer. Bounds ≈ 2× SENSITIVITY; CONTROL is 0.38.
      statsTol := if bf16 then 3e-2 else 5e-3, dupTol }
  | "resnet50bce" => some
    { slug := s!"resnet50in160bce{p}", net := resnet50Imagenet160Verified.toNet, bs := 32
      replicas := 4, sgPath := "", dpPath := ""
      render := fun B fs => resnet50TrainStepFaithfulB B 1000 "1.0e-05" 1 .lamb
        "resnet50in160" (bce := true) (q := 5) (bf16 := bf16) (forceSync := fs)
      renderDp := fun B R => resnet50TrainStepFaithfulB B 1000 "1.0e-05" R .lamb
        "resnet50in160" (bce := true) (q := 5) (bf16 := bf16)
      entry := fun B r =>
        s!"m.resnet50in160_{r34AdamVariant B r .lamb false false true "" bf16}_train_step"
      shardShift := 0.5
      -- resnet50's bounds: TEST 1.08 / 1.09 / 1.22e-2 bf16 and 2.2e-3 f32, SENSITIVITY 1.5e-2 / 1.7e-3
      statsTol := if bf16 then 3e-2 else 5e-3, dupTol }
  | "mnv4" => some
    { slug := s!"mnv4in{p}", net := mnv4ImagenetVerified.toNet, bs := 64, replicas := 4
      sgPath := ""
      dpPath := s!"verified_mlir/mnv4in_adamdp64{p}_train_step.mlir"
      render := fun B fs => mobilenetv4AdamTrainStepFaithfulB B 1000 "1.0e-5" 1 "mnv4in" bf16
        (forceSync := fs)
      entry := fun B r => s!"m.mnv4in_{mnv4AdamVariant B r bf16}_train_step"
      shardShift := 0.5
      -- 77 layers: statistics TEST 6.9e-3 / 1.16e-2 bf16 and 1.2e-3 / 1.3e-3 f32 on two runs
      -- each, SENSITIVITY 1.6e-2 / 1.3e-3; per-layer variance error grows smoothly to 1.0e-2 /
      -- 1.7e-3 and the first four layers are split-exact. Bounds ≈ 2× SENSITIVITY; CONTROL 0.076.
      statsTol := if bf16 then 3e-2 else 3e-3, dupTol }
  | _ => none

def main (args : List String) : IO Unit := do
  let net := args.head?.getD "resnet34"
  let bf16 := !args.contains "f32"
  match imagenetCfg net bf16 with
  | some cfg => SyncBnCheck.run cfg
  | none =>
    IO.eprintln s!"unknown net '{net}': resnet34 | mobilenetv2 | efficientnet | resnet50 | resnet50bce | mnv4 [f32]"
    IO.Process.exit 2
