import LeanMlir

/-! Emit the standalone FPN-neck (top-down multi-scale merge) module for numeric
    validation (detection-infra brick #3, planning/yolo_fpn.md bite 2). Writes
    `fpn_neck_gen.mlir` for a concrete (B, oc, c3, c4, c5, g5);
    `scripts/fpn_neck_probe_check.py` compiles it with IREE (CPU) and checks the
    emitted forward against the numpy `fpn_forward` and the emitted backward
    (dC3/dC4/dC5 + dW3/dW4/dW5) against the f64-FD-verified `fpn_grad` oracle in
    `scripts/fpn_neck_check.py`.

    Usage: lake exe fpn-neck-probe [B oc c3 c4 c5 g5] [outPath]
    Defaults: 2 8 6 10 12 3 fpn_neck_gen.mlir -/
def main (args : List String) : IO Unit := do
  let nums := args.filterMap String.toNat?
  let n (i d : Nat) : Nat := (nums[i]?).getD d
  let B := n 0 2; let oc := n 1 8
  let c3 := n 2 6; let c4 := n 3 10; let c5 := n 4 12
  let g5 := n 5 3
  let outPath := (args.filter (·.endsWith ".mlir")).head?.getD "fpn_neck_gen.mlir"
  let mlir := MlirCodegen.fpnNeckProbeModule B oc c3 c4 c5 g5
  IO.FS.writeFile outPath mlir
  IO.eprintln s!"wrote {outPath}  (FPN neck probe B={B} oc={oc} c3={c3} c4={c4} c5={c5} g5={g5}, {mlir.length} chars)"
