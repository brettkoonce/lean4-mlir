import LeanMlir.Proofs.Codegen.ResNet50RenderB
import LeanMlir.VerifiedNets

/-! # The R50 render ↔ driver parameter contract, run rather than reasoned about

    lake env lean tests/TestR50Contract.lean

**What this gates.** The renderer (`ResNet50RenderB.r50ShapeList`) and the driver
(`VerifiedNets.resnet50ImagenetVerified.toSpecs`, via `VerifiedSpec.bottleneckStageSpec`) each
carry a list of R50's parameter tensors. The driver packs `[θ|m|v]` off ITS list and hands the blob
to a graph laid out by the OTHER one. If the two disagree in count, in order, or in any single
shape, every parameter after the first divergence is fed to the wrong slot — and the run does not
crash, it trains a scrambled net and reports a loss curve. `resnet34in`'s conv-bias slot (§2m) is
the standing precedent: one tensor in the wrong place, silent until the driver mis-walked the blob.

⚠ These are two independent definitions of one fact and they cannot be merged: the spec's exists to
drive initialisation and checkpointing, the render's to emit MLIR text. So the honest move is to
diff them, which is what this does.

**The three checks, weakest to strongest:**

1. count — 161 tensors
2. total parameters — **25,557,032**, which is torchvision's ResNet-50 and the reference's own
   reported count. ⚠ This one is a *published* number rather than an internal one, so it also
   catches both lists being wrong in the same way.
3. ⭐ elementwise shape equality, in order — the only one that actually pins the packing.

⚠ Check 3 subsumes 1 and 2; they are kept because when 3 fails, which of them ALSO fails says
immediately whether the fault is a missing tensor, a mis-sized one, or a permutation.
-/

open Proofs.StableHLO

/-- The renderer's parameter shapes, in emitted func-arg order. -/
def renderShapes : List (List Nat) := (r50ShapeList 1000).map (·.2)

/-- The spec's parameter shapes, in the order the driver initialises and packs them. -/
def specShapes : List (List Nat) :=
  (resnet50ImagenetVerified.toSpecs.map (fun (ds, _) => ds.toList)).toList

private def total (ls : List (List Nat)) : Nat := ls.foldl (fun a d => a + d.foldl (·*·) 1) 0

-- 1. Tensor count. 3 stem + (12+9+9) + (12+27) + (12+45) + (12+18) + 2 head.
#guard renderShapes.length == 161
#guard specShapes.length == 161

-- 2. ⭐ The published parameter count. `planning/rsb_a3_r50_verified.md` §1 measured this exact on
--    the first try from the layout spec alone; this pins the RENDER to it too.
#guard total renderShapes == 25557032
#guard total specShapes == 25557032

-- 3. ⭐⭐ The real gate: same shapes, same order, tensor for tensor.
#guard renderShapes == specShapes

-- The BN running-stat contract, the same argument one level over: the driver packs
-- `runningBnStats` as `bnChannels.foldl (fun acc c => acc ++ #[#[c], #[c]])`, i.e. μ and var
-- interleaved per layer, so the render's stat slot list must be exactly twice `bnChannels` long
-- and agree with it channel for channel.
#guard r50StatSigList.length == 2 * resnet50ImagenetVerified.bnChannels.size

def main : IO Unit := do
  IO.println s!"R50 render ↔ driver contract"
  IO.println s!"  tensors      render {renderShapes.length}  spec {specShapes.length}"
  IO.println s!"  parameters   render {total renderShapes}  spec {total specShapes}  (want 25557032)"
  IO.println s!"  BN stat slots {r50StatSigList.length}  = 2 x {resnet50ImagenetVerified.bnChannels.size} BN layers"
  if renderShapes == specShapes then
    IO.println "  ✓ shapes agree elementwise, in order"
  else
    IO.println s!"  ✗ FIRST MISMATCH: {(renderShapes.zip specShapes).find? (fun (a, b) => a != b)}"
    throw <| IO.userError "R50 render/spec parameter contract broken"
