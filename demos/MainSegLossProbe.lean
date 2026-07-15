import LeanMlir

/-! Emit the standalone segmentation-loss module for FD validation
    (planning/brats_demo.md Workstream B). Writes `seg_loss_gen.mlir` for a
    concrete (B, NC, H, W) and loss kind; `scripts/seg_loss_probe_check.py`
    compiles it with IREE and checks `d_logits` against central finite
    differences of `loss`.

    Dice is the reason this exists. Per-pixel CE's backward seed collapses to
    `(softmax - onehot)/N` — the softmax Jacobian cancels against the log, so
    it is verifiable by inspection. Dice gets no such cancellation and carries
    an explicit Jacobian-vector product, so it wants a numeric check.

    Usage: lake exe seg-loss-probe [ce|dice|dicece] [B NC H W] [ls=F] [outPath]
    Defaults: dice 2 4 3 3 ls=0 seg_loss_gen.mlir -/

def main (args : List String) : IO Unit := do
  let nums := (args.filterMap String.toNat?)
  let n (i : Nat) (d : Nat) : Nat := (nums[i]?).getD d
  let segLoss : SegLoss :=
    if args.any (· == "ce") then .ce
    else if args.any (· == "dicece") then .diceCE
    else .dice
  let ls : Float :=
    match (args.filter (·.startsWith "ls=")).head? with
    | some a => ((a.drop 3).toNat?.map Nat.toFloat).getD 0.0 / 100.0
    | none => 0.0
  let B := n 0 2; let NC := n 1 4; let H := n 2 3; let W := n 3 3
  let outPath := (args.filter (fun a => a.endsWith ".mlir")).head?.getD "seg_loss_gen.mlir"
  let mlir := MlirCodegen.segLossProbeModule B NC H W segLoss ls
  IO.FS.writeFile outPath mlir
  IO.eprintln s!"wrote {outPath}  (loss={repr segLoss} B={B} NC={NC} H={H} W={W} ls={ls}, {mlir.length} chars)"
