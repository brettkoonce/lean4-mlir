import LeanMlir

/-! Emit the standalone FlashAttention-forward module for validation
    (planning/flash_attention.md rung 2–3). Writes `flash_gen.mlir` for a
    concrete (b, heads, n, dh, bk); a companion Python harness compiles it
    with IREE and checks the output against dense attention.

    Usage: lake exe flash-probe [b heads n dh bk] [causal] [outPath]
    Defaults: 1 2 16 8 4  full  flash_gen.mlir -/

def main (args : List String) : IO Unit := do
  let n (i : Nat) (d : Nat) : Nat := ((args[i]?).bind String.toNat?).getD d
  let b := n 0 1; let heads := n 1 2; let seqN := n 2 16; let dh := n 3 8; let bk := n 4 4
  let causal := args.any (· == "causal")
  let bwd := args.any (· == "bwd")   -- emit the fwd+bwd module instead of fwd-only
  let dense := args.any (· == "dense")  -- emit the dense-SDPA baseline (memory comparison)
  let rope := args.any (· == "rope")    -- emit a standalone RoPE module
  let outPath := (args.filter (fun a => a.endsWith ".mlir")).head?.getD "flash_gen.mlir"
  let mlir := if rope then MlirCodegen.ropeProbeModule b heads seqN dh bwd
              else if dense then MlirCodegen.denseSdpaProbeModule b heads seqN dh causal
              else if bwd then MlirCodegen.flashBwdProbeModule b heads seqN dh bk causal
              else MlirCodegen.flashProbeModule b heads seqN dh bk causal
  IO.FS.writeFile outPath mlir
  IO.eprintln s!"wrote {outPath}  (b={b} heads={heads} n={seqN} dh={dh} bk={bk} causal={causal} bwd={bwd}, {mlir.length} chars)"
