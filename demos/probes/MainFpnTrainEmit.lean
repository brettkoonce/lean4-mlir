import LeanMlir
import LeanMlir.ReferenceNets

/-! Emit-only harness for the FPN multi-scale detector train step. Constructs the
    `r34FpnDet` spec (R34-ImageNet backbone tapped at C3/C4/C5 → `.fpnDetect` →
    flat [B,Ntot]) and dumps the generated train-step MLIR so it can be eyeballed
    / parse-checked with `iree-compile --compile-to=input` BEFORE the ~15-min ROCm
    compile.

    Usage: lake exe fpn-train-emit [outDir]     (head tower depth via FPN_TOWER)
-/

-- Per-scale k-means anchor placeholders (real priors in data/visdrone/anchors_fpn_*.txt).
def fpnAnchorsP3 : List (Float × Float) := [(0.007, 0.015), (0.017, 0.027), (0.023, 0.061)]
def fpnAnchorsP4 : List (Float × Float) := [(0.044, 0.038), (0.062, 0.087), (0.090, 0.120)]
def fpnAnchorsP5 : List (Float × Float) := [(0.124, 0.151), (0.200, 0.220), (0.300, 0.400)]

def fpnDetScales : List (Nat × List (Float × Float)) :=
  [(56, fpnAnchorsP3), (28, fpnAnchorsP4), (14, fpnAnchorsP5)]

def r34FpnDetT (tower : Nat) : NetSpec :=
  ReferenceNets.r34FpnDet "ResNet-34 + FPN detector 448 (VisDrone)" tower

def main (args : List String) : IO Unit := do
  let outDir := args.head?.getD "/tmp"
  let tower := ((← IO.getEnv "FPN_TOWER").bind (·.trimAscii.toNat?)).getD 0
  let r34FpnDet := r34FpnDetT tower
  let batch := 8
  let train := MlirCodegen.generateTrainStep r34FpnDet batch "jit_fpn_train_step"
    (useAdam := true) (weightDecay := 0.0005) (gradClipNorm := 4.0)
    (focalGamma := 2.0) (fpnScales := fpnDetScales)
  let fwd  := MlirCodegen.generate r34FpnDet batch
  let eval := MlirCodegen.generateEval r34FpnDet batch
  IO.FS.writeFile s!"{outDir}/fpn_train_step.mlir" train
  IO.FS.writeFile s!"{outDir}/fpn_fwd.mlir" fwd
  IO.FS.writeFile s!"{outDir}/fpn_fwd_eval.mlir" eval
  let ntot := (fpnDetScales.map (fun sc => sc.2.length * 15 * sc.1 * sc.1)).foldl (·+·) 0
  IO.println s!"train={train.length}  fwd={fwd.length}  eval={eval.length} chars -> {outDir}  (Ntot = {ntot}, tower = {tower}, params = {r34FpnDet.totalParams})"
