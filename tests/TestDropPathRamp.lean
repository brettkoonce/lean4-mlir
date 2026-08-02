import LeanMlir.VerifiedNets
import LeanMlir.Proofs.Codegen.EfficientNetRender

/-! # The stochastic-depth ramp, pinned across the driver/renderer seam

`planning/stochastic_depth.md`'s keep ramp is described **twice**, and it has to be:

* the RENDERER (`Proofs/Codegen/EfficientNetRender.lean`) owns `enetDropIdxs` / `enetDropTotal` —
  which blocks carry a drop site, and the ramp denominator;
* the DRIVER (`efficientnetVerified.dropKeeps`) owns the keep VALUES, because `1/keep_i` is folded
  into the supplied scale rather than baked into the graph (`Proofs.dropPath`'s note has the
  argument: a baked constant and "the forward emits the sites too" cannot both hold).

`LeanMlir/VerifiedSpec.lean` imports `VerifiedTrain`, so the renderer cannot share the definition by
import without inverting the dependency — the driver would drag the whole proof corpus into the app
build. That is exactly the situation `#guard spec.toSpecs == XLayout.specs` already handles for the
parameter layout, and this file is its peer for the ramp.

**What it catches.** A ramp re-indexed by SITE ORDINAL instead of BLOCK INDEX — nine evenly-spaced
keeps instead of the reference's nine uneven ones. That compiles, runs, descends, and trains a
different objective, and **no numeric tie can see it**: every tie compares the render against a peer
built from the same constants. §2k's `α/K` bug in a new place.

    lake env lean tests/TestDropPathRamp.lean
-/

open Proofs.StableHLO

/-- The reference's ramp, restated here from `jax/Jax/Codegen.lean:2031-2040` rather than imported
    from either side — so this is a third reading, not a copy of one of the two under test. -/
private def refKeep (dropRate : Float) (i totalDrop : Nat) : Float :=
  1.0 - dropRate * i.toFloat / (Nat.max 1 (totalDrop - 1)).toFloat

-- ⭐ The driver's keep values ARE the reference's ramp at the renderer's site indices.
#guard efficientnetVerified.dropKeeps.size == enetDropIdxs.length

#guard (enetDropIdxs.toArray.zip efficientnetVerified.dropKeeps).all
         (fun (i, k) => ((k - refKeep 0.2 i enetDropTotal).abs) < 1e-9)

-- The site count the renderer emits and the count the driver supplies must agree, or the graph
-- gets the wrong number of `tensor<Bxf32>` inputs — a loud arity failure, but only at run time.
#guard enetDropSites == efficientnetVerified.dropKeeps.size

-- ⚠ Block 0 keeps everything, and the LAST block would keep `1 − dropRate`; but EfficientNet's
-- last block (15) has NO SKIP, so the deepest site actually rendered is block 14 at keep 0.8133…,
-- NOT 0.8. Stated because "the last site keeps 1 − dropRate" is the natural wrong assumption, and
-- here it is off by exactly one ramp step.
#guard (refKeep 0.2 0 16 - 1.0).abs < 1e-9
#guard (efficientnetVerified.dropKeeps[8]! - refKeep 0.2 14 16).abs < 1e-9
#guard (efficientnetVerified.dropKeeps[8]! - 0.8).abs > 1e-3

#eval do
  IO.println "── stochastic-depth ramp, driver vs renderer ──"
  for (i, k) in enetDropIdxs.toArray.zip efficientnetVerified.dropKeeps do
    IO.println s!"  block {i}: keep {k}"
  IO.println s!"✓ {enetDropSites} sites, denominator {enetDropTotal - 1}, ramp agrees with the reference"
