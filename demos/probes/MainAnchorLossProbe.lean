import LeanMlir

/-! Emit the standalone anchor-YOLO-loss module for numeric validation (brick #2,
    WS-C). Writes `anchor_loss_gen.mlir` for a concrete (B, gH, gW, A);
    `scripts/anchor_loss_probe_check.py` compiles it (CPU) and checks the emitted
    forward against numpy and the emitted backward against finite differences.

    Usage: lake exe anchor-loss-probe [B gH gW A] [outPath]
    Defaults: 2 5 5 3 anchor_loss_gen.mlir -/
def main (args : List String) : IO Unit := do
  let nums := args.filterMap String.toNat?
  let n (i d : Nat) : Nat := (nums[i]?).getD d
  let B := n 0 2; let gH := n 1 5; let gW := n 2 5; let A := n 3 3
  let outPath := (args.filter (·.endsWith ".mlir")).head?.getD "anchor_loss_gen.mlir"
  let mlir := MlirCodegen.anchorLossProbeModule B gH gW A
  IO.FS.writeFile outPath mlir
  IO.eprintln s!"wrote {outPath}  (anchor-loss probe B={B} gH={gH} gW={gW} A={A}, {mlir.length} chars)"
