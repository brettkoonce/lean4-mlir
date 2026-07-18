import LeanMlir

/-! Emit the standalone whole-FPN-detector module (neck + 1×1 heads + concat +
    multi-scale loss + full DAG backward) for numeric validation (planning/
    yolo_fpn.md bite 7 de-risk). Writes `fpn_detect_gen.mlir`;
    `scripts/fpn_detect_probe_check.py` compiles it with IREE (CPU) and f64-FD-
    checks every input/param gradient. Emitted at focal γ=0 so the objectness
    weight is a genuine constant and the whole loss is exactly differentiable.
    Conv1x1-only ⇒ CPU-compiles (the real detector's conv backbone is ROCm-only).

    Usage: lake exe fpn-detect-probe [B oc c3 c4 c5 g5 A] [outPath]
    Defaults: 2 8 6 10 12 2 3 fpn_detect_gen.mlir -/
def main (args : List String) : IO Unit := do
  let nums := args.filterMap String.toNat?
  let n (i d : Nat) : Nat := (nums[i]?).getD d
  let B := n 0 2; let oc := n 1 8
  let c3 := n 2 6; let c4 := n 3 10; let c5 := n 4 12
  let g5 := n 5 2; let A := n 6 3
  let outPath := (args.filter (·.endsWith ".mlir")).head?.getD "fpn_detect_gen.mlir"
  let mlir := MlirCodegen.fpnDetectProbeModule B oc c3 c4 c5 g5 A
  IO.FS.writeFile outPath mlir
  IO.eprintln s!"wrote {outPath}  (FPN detect probe B={B} oc={oc} c=({c3},{c4},{c5}) g5={g5} A={A}, {mlir.length} chars)"
