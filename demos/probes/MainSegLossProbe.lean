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

    Usage: lake exe seg-loss-probe [ce|dice|dicece|wce|wce1|focal|focal0]
                                   [B NC H W] [ls=F] [outPath]
    Defaults: dice 2 4 3 3 ls=0 seg_loss_gen.mlir

    `wce1` and `focal0` are the degenerate-parameter twins of `ce` (all-ones
    weights; γ=0). Both must reproduce it exactly — a check finite differences
    cannot make, since FD only asks that the gradient match whatever loss was
    emitted, not that it is the loss we meant. -/

/-- Deterministic weight vector for the `wce` probe: `w_c = 1 + c`.

    Non-uniform (so the weighting actually does something), small integers (so
    they are exact in f32 and any FD discrepancy is the emitter's fault, not the
    constant's), and a function of `NC` alone — which is what lets
    `seg_loss_probe_check.py` mirror it without teaching the CLI to parse
    floats. -/
def probeWeights (NC : Nat) : List Float :=
  (List.range NC).map (fun c => 1.0 + c.toFloat)

/-- Minimal non-negative decimal parser. Lean core has no `String.toFloat?`, and
    the alternative — baking the demo's weight vector into this file too — is a
    constant duplicated across two modules with no way to keep them honest.
    Handles `"3"`, `"3.5"`, `"220.0868"`; rejects anything else, which is the
    right answer for a probe CLI. -/
private def parseFloat? (s : String) : Option Float :=
  match s.splitOn "." with
  | [a] => a.toNat?.map Nat.toFloat
  | [a, b] => do
    let ai ← a.toNat?
    let bi ← b.toNat?
    let scale := (List.range b.length).foldl (fun acc _ => acc * 10.0) 1.0
    some (ai.toFloat + bi.toFloat / scale)
  | _ => none

def main (args : List String) : IO Unit := do
  let nums := (args.filterMap String.toNat?)
  let n (i : Nat) (d : Nat) : Nat := (nums[i]?).getD d
  let ls : Float :=
    match (args.filter (·.startsWith "ls=")).head? with
    | some a => ((a.drop 3).toNat?.map Nat.toFloat).getD 0.0 / 100.0
    | none => 0.0
  let B := n 0 2; let NC := n 1 4; let H := n 2 3; let W := n 3 3
  -- `w=1.0:60.9033:...` overrides `wce`'s weights; `g=2.0` overrides `focal`'s
  -- γ. Both default to the deterministic probe values, so the FD checks that
  -- pass no override keep testing exactly what they always did.
  -- `.drop` yields a `String.Slice` in this toolchain; `.toString` back before
  -- reaching for the String API.
  let cliWeights : Option (List Float) :=
    match (args.filter (·.startsWith "w=")).head? with
    | some a => (((a.drop 2).toString).splitOn ":").mapM parseFloat?
    | none => none
  let gamma : Float :=
    match (args.filter (·.startsWith "g=")).head? with
    | some a => (parseFloat? (a.drop 2).toString).getD 2.0
    | none => 2.0
  -- `wce1` and `focal0` are the degenerate-parameter twins of `ce`: an all-ones
  -- weight vector, and γ=0. Each must reproduce plain CE exactly — the cheapest
  -- test that the Σw denominator and focal's A-sign are right, and one FD
  -- cannot make on its own.
  let segLoss : SegLoss :=
    if args.any (· == "ce") then .ce
    else if args.any (· == "dicece") then .diceCE
    else if args.any (· == "wce1") then .weightedCE (List.replicate NC 1.0)
    else if args.any (· == "wce") then .weightedCE (cliWeights.getD (probeWeights NC))
    else if args.any (· == "focal0") then .focalCE 0.0
    else if args.any (· == "focal") then .focalCE gamma
    else .dice
  let outPath := (args.filter (fun a => a.endsWith ".mlir")).head?.getD "seg_loss_gen.mlir"
  let mlir := MlirCodegen.segLossProbeModule B NC H W segLoss ls
  IO.FS.writeFile outPath mlir
  IO.eprintln s!"wrote {outPath}  (loss={repr segLoss} B={B} NC={NC} H={H} W={W} ls={ls}, {mlir.length} chars)"
