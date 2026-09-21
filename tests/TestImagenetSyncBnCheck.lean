import LeanMlir.SyncBnCheck
import LeanMlir.Proofs.Codegen.ResNet34RenderB
import LeanMlir.Proofs.Codegen.MobileNetV2RenderB
import LeanMlir.Proofs.Codegen.EfficientNetRender

/-! # `imagenet-syncbn-check` — synchronised BatchNorm at the ImageNet shape: 4×64 IS 1×256

    lake build imagenet-syncbn-check
    unset HIP_VISIBLE_DEVICES
    PJRT_REPLICAS=4 .lake/build/bin/imagenet-syncbn-check resnet34        # momdp64bf16
    PJRT_REPLICAS=4 .lake/build/bin/imagenet-syncbn-check mobilenetv2     # rmsdp64bf16
    PJRT_REPLICAS=4 .lake/build/bin/imagenet-syncbn-check efficientnet    # rmsdp64bf16
    LEAN_MLIR_MEM_FRACTION=0.97 PJRT_REPLICAS=4 … <net> f32               # the f32 peer

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
  | _ => none

def main (args : List String) : IO Unit := do
  let net := args.head?.getD "resnet34"
  let bf16 := !args.contains "f32"
  match imagenetCfg net bf16 with
  | some cfg => SyncBnCheck.run cfg
  | none =>
    IO.eprintln s!"unknown net '{net}': resnet34 | mobilenetv2 | efficientnet [f32]"
    IO.Process.exit 2
