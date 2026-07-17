import LeanMlir

/-! Emit the standalone FPN multi-scale-loss module for numeric validation
    (detection-infra brick #3, planning/yolo_fpn.md bites 4+6). Writes
    `fpn_loss_gen.mlir` for concrete (B, A, 3 grids); `scripts/fpn_loss_probe_check.py`
    compiles it with IREE (CPU) and checks the emitted forward+backward against an
    independent numpy multi-scale reference (Σ of per-scale anchor losses) and its
    f64 finite differences. Conv-free, so it CPU-compiles (the conv heads feeding
    this in the real detector are verified convBn, validated on ROCm).

    Usage: lake exe fpn-loss-probe [B A g3 g4 g5] [outPath]
    Defaults: 2 3 8 4 2 fpn_loss_gen.mlir -/
def main (args : List String) : IO Unit := do
  let nums := args.filterMap String.toNat?
  let n (i d : Nat) : Nat := (nums[i]?).getD d
  let B := n 0 2; let A := n 1 3
  let g3 := n 2 8; let g4 := n 3 4; let g5 := n 4 2
  let outPath := (args.filter (·.endsWith ".mlir")).head?.getD "fpn_loss_gen.mlir"
  let mlir := MlirCodegen.fpnLossProbeModule B [g3, g4, g5] A
  IO.FS.writeFile outPath mlir
  IO.eprintln s!"wrote {outPath}  (FPN loss probe B={B} A={A} grids=[{g3},{g4},{g5}], {mlir.length} chars)"
