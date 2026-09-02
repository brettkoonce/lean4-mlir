import LeanMlir

/-! Emit the standalone DIoU box-loss forward module for numeric validation
    (brick #1, planning/yolo_drone.md WS-D). Writes `diou_loss_gen.mlir` for a
    concrete (B, gH, gW); `scripts/diou_probe_check.py` compiles it with IREE
    (CPU) and checks the emitted `loss` against the numpy reference in
    `scripts/diou_grad_check.py`. Chunk 2a — forward only; the backward VJP is
    the next probe.

    Usage: lake exe diou-loss-probe [B gH gW] [outPath]
    Defaults: 2 14 14 diou_loss_gen.mlir -/
def main (args : List String) : IO Unit := do
  let nums := args.filterMap String.toNat?
  let n (i d : Nat) : Nat := (nums[i]?).getD d
  let B := n 0 2; let gH := n 1 14; let gW := n 2 14
  -- anchor priors as integer permille (aw=50 → 0.050); default 1.0 = anchor-free.
  let permille := fun (pfx : String) (d : Float) =>
    match (args.filter (·.startsWith pfx)).head?.bind (fun a => (a.drop pfx.length).toString.toNat?) with
    | some v => v.toFloat / 1000.0
    | none => d
  let anchorW := permille "aw=" 1.0
  let anchorH := permille "ah=" 1.0
  let outPath := (args.filter (·.endsWith ".mlir")).head?.getD "diou_loss_gen.mlir"
  let mlir := MlirCodegen.diouProbeModule B gH gW anchorW anchorH
  IO.FS.writeFile outPath mlir
  IO.eprintln s!"wrote {outPath}  (DIoU probe B={B} gH={gH} gW={gW} anchorW={anchorW} anchorH={anchorH}, {mlir.length} chars)"
