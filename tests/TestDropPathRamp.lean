import LeanMlir.VerifiedNets
import LeanMlir.Proofs.Codegen.EfficientNetRender
import LeanMlir.Proofs.Codegen.ConvNeXtRenderB
import LeanMlir.Proofs.Codegen.ViTRenderB

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

-- ══════════════════════════════════════════════════════════════════════════════════════════════
--  ▶ ConvNeXt-T — the same seam, and the trap is a DIFFERENT one
--
--  EfficientNet's hazard is the SITE ORDINAL (9 sites, 16 blocks, so ordinal ≠ block index).
--  ConvNeXt has one site per block, so that particular confusion is impossible — and the hazard
--  moves to the STAGE. `cDepths` is `[3,3,9,3]`, four stages, and the reference's `dbi` is ONE
--  counter across all of them: stage 1's first block is ramp index 3, not 0. A per-stage index
--  gives four short ramps (keeps 1.0, 0.967, 0.933 repeated) instead of one long one — same site
--  count, same arity, same emitted op count, different objective.
-- ══════════════════════════════════════════════════════════════════════════════════════════════

-- ⭐ The driver's keeps ARE the reference's ramp at the renderer's ramp indices, all 18.
#guard convnextVerified.dropKeeps.size == cnxDropSites

#guard ((Array.range cnxDropSites).zip convnextVerified.dropKeeps).all
         (fun (i, k) => ((k - refKeep 0.1 i cnxDropTotal).abs) < 1e-9)

-- ⚠ THE STAGE BOUNDARIES, which is where the ConvNeXt-specific defect would show. Read through
-- `cnxBlockIdx` — the renderer's own numbering — rather than restated, so a change there fails here.
#guard (convnextVerified.dropKeeps[cnxBlockIdx 1 0]! - refKeep 0.1 3 18).abs < 1e-9
#guard (convnextVerified.dropKeeps[cnxBlockIdx 2 0]! - refKeep 0.1 6 18).abs < 1e-9
#guard (convnextVerified.dropKeeps[cnxBlockIdx 3 0]! - refKeep 0.1 15 18).abs < 1e-9
-- …and the per-stage misreading must NOT agree with it, or the guards above are vacuous.
#guard (convnextVerified.dropKeeps[cnxBlockIdx 3 0]! - refKeep 0.1 0 18).abs > 1e-3

-- ⚠ Block 0 keeps EXACTLY 1.0 (the reference's `keep_prob < 1.0` guard, obtained as data), and the
-- last block keeps exactly `1 − dropRate` — unlike EfficientNet, whose deepest SITE is one ramp
-- step short of that because its last block carries no skip.
#guard convnextVerified.dropKeeps[0]! == 1.0
#guard (convnextVerified.dropKeeps[17]! - 0.9).abs < 1e-9

-- Both scales carry the same ramp: it is a property of the architecture and of `dropPath := 0.1`,
-- not of the class count. §0.4 finding 5 — a feature is not done when its Imagenette artifact
-- renders — with the check that says so rather than the comment.
#guard convnextImagenetVerified.dropKeeps == convnextVerified.dropKeeps

-- ══════════════════════════════════════════════════════════════════════════════════════════════
--  ▶ ConvNeXt-**S** — the same seam again, and now the SIZE is what moves
--
--  ⚠⚠ Two things change together here and only one of them is architectural. The site COUNT goes
--  18 → 36 because stage 3 deepens (that follows from `cnxSmall`, and the renderer derives it).
--  The RATE goes 0.1 → 0.4 because the ConvNeXt paper sets stochastic depth per model size, and
--  NOTHING derives that — it is a recipe number this spec asserts. So the two guards below are of
--  genuinely different kinds: the first says the renderer and the spec agree about the net, the
--  second says the spec did not silently inherit Tiny's regularisation strength.
-- ══════════════════════════════════════════════════════════════════════════════════════════════

-- One spec entry per RENDERED site, read through the renderer's own depth table.
#guard convnextSImagenetVerified.dropKeeps.size == cnxDropSites cnxSmall
#guard convnextSImagenetVerified.dropKeeps.size == 36

-- The renderer's parameter list and the spec's layer list must be the same 342 tensors — the
-- `toSpecs == Layout.specs` move (§2m) carried across the depth parameter. This is what fires if
-- `cnxSmall` and the spec's layer list ever disagree about WHICH stage deepens: both would still
-- have 36 blocks and 342 tensors, but the shapes would differ.
#guard (cnxAllParams 1000 cnxSmall).map (fun (_, ds) => ds.toArray) ==
         (convnextSImagenetVerified.toSpecs.map (fun (d, _) => d)).toList

-- The ramp itself, at rate 0.4 over 36 sites, through `cnxBlockIdx` at the S table. ⚠ Stage 3
-- starts at 33 here and at 15 for Tiny — passing `cnxSmall` is the whole content of these lines,
-- and omitting it silently takes the T default (the ViT-S trap, mechanised).
#guard ((Array.range (cnxDropSites cnxSmall)).zip convnextSImagenetVerified.dropKeeps).all
         (fun (i, k) => ((k - refKeep 0.4 i (cnxDropTotal cnxSmall)).abs) < 1e-9)
#guard (convnextSImagenetVerified.dropKeeps[cnxBlockIdx 2 0 cnxSmall]! - refKeep 0.4 6 36).abs < 1e-9
#guard (convnextSImagenetVerified.dropKeeps[cnxBlockIdx 3 0 cnxSmall]! - refKeep 0.4 33 36).abs < 1e-9
-- …and Tiny's stage-3 start must NOT agree with it, or the line above is vacuous.
#guard (convnextSImagenetVerified.dropKeeps[cnxBlockIdx 3 0]! - refKeep 0.4 33 36).abs > 1e-3

-- ⭐ THE RATE IS NOT INHERITED. Block 0 still keeps exactly 1.0, but the deepest block keeps 0.6
-- where Tiny's keeps 0.9 — the ConvNeXt paper's per-size stochastic depth (S 0.4, T 0.1). A guard
-- on the ramp's SIZE alone passes on a copied Tiny ramp; this one does not.
#guard convnextSImagenetVerified.dropKeeps[0]! == 1.0
#guard (convnextSImagenetVerified.dropKeeps[35]! - 0.6).abs < 1e-9
#guard (convnextSImagenetVerified.dropKeeps[35]! - convnextVerified.dropKeeps[17]!).abs > 1e-3

-- ══════════════════════════════════════════════════════════════════════════════════════════════
--  ▶ ConvNeXt-**B** — and here the seam is that NOTHING about the ramp moves
--
--  ⚠⚠ B shares S's depth table exactly, so the site COUNT, the indices and the ramp SHAPE are
--  identical. The only thing that differs is the RATE (0.5 vs 0.4), which no gate can derive. So
--  the two guards below are: "B's sites really are S's sites" (checked through the renderer's own
--  `cnxBlockIdx` at B's record) and "B's rate really is not S's" — and it is the second that a
--  copy-paste of the S spec would fail, since the first would pass on it.
-- ══════════════════════════════════════════════════════════════════════════════════════════════

#guard convnextBImagenetVerified.dropKeeps.size == cnxDropSites cnxBase
#guard cnxBlockIdx 3 0 cnxBase == cnxBlockIdx 3 0 cnxSmall
#guard ((Array.range (cnxDropSites cnxBase)).zip convnextBImagenetVerified.dropKeeps).all
         (fun (i, k) => ((k - refKeep 0.5 i (cnxDropTotal cnxBase)).abs) < 1e-9)
#guard convnextBImagenetVerified.dropKeeps[0]! == 1.0
#guard (convnextBImagenetVerified.dropKeeps[35]! - 0.5).abs < 1e-9

-- The renderer's parameter list == the spec's layer list, at B's widths. ⚠ This is the guard that
-- would fire if the spec's hand-written layer list and `cnxBase.dims` disagreed — and unlike S's
-- peer it is load-bearing, because B is where those widths are written out by hand in two places.
#guard (cnxAllParams 1000 cnxBase).map (fun (_, ds) => ds.toArray) ==
         (convnextBImagenetVerified.toSpecs.map (fun (d, _) => d)).toList

-- ══════════════════════════════════════════════════════════════════════════════════════════════
--  ▶ ViT-Tiny — a THIRD shape of the same seam, and the trap moves again
--
--  EfficientNet's hazard is the SITE ORDINAL (9 sites over 16 blocks). ConvNeXt's is the STAGE
--  (one counter across four of them). ViT's is that **sites ≠ ramp index in the other direction**:
--  24 sites over 12 blocks, TWO per block, both at the SAME keep. Deriving the ramp from the site
--  ordinal gives 24 evenly-spaced keeps where the reference has 12 PAIRS — same site count, same
--  arity, same emitted op count, different objective.
-- ══════════════════════════════════════════════════════════════════════════════════════════════

#guard vitVerified.dropKeeps.size == vitDropSites
#guard vitDropSites == 2 * vitDropTotal

-- ⭐ Every site's keep is the reference's ramp AT ITS BLOCK INDEX, read through the renderer's own
-- `vitRampOf` rather than restated.
#guard ((Array.range vitDropSites).zip vitVerified.dropKeeps).all
         (fun (sIdx, k) => ((k - refKeep 0.1 (vitRampOf sIdx) vitDropTotal).abs) < 1e-9)

-- ⚠ THE PAIRING, which is where the ViT-specific defect would show: the two branches of a block
-- share one keep…
#guard (List.range vitDropTotal).all (fun i =>
         vitVerified.dropKeeps[vitSiteIdx i 0]! == vitVerified.dropKeeps[vitSiteIdx i 1]!)
-- …and consecutive BLOCKS do not, or the check above is vacuous.
#guard (List.range (vitDropTotal - 1)).all (fun i =>
         vitVerified.dropKeeps[vitSiteIdx i 0]! != vitVerified.dropKeeps[vitSiteIdx (i+1) 0]!)
-- ⚠ And the site-ordinal misreading must NOT agree with it: at site 2 the correct keep is block 1's,
-- not site 2's.
#guard (vitVerified.dropKeeps[2]! - refKeep 0.1 2 vitDropTotal).abs > 1e-3

-- Block 0 keeps exactly 1.0; block 11 keeps exactly `1 - dropPath`.
#guard vitVerified.dropKeeps[0]! == 1.0
#guard (vitVerified.dropKeeps[23]! - 0.9).abs < 1e-9

#guard vitImagenetVerified.dropKeeps == vitVerified.dropKeeps

#eval do
  IO.println "── stochastic-depth ramp, driver vs renderer ──"
  IO.println "EfficientNet-B0:"
  for (i, k) in enetDropIdxs.toArray.zip efficientnetVerified.dropKeeps do
    IO.println s!"  block {i}: keep {k}"
  IO.println s!"✓ {enetDropSites} sites, denominator {enetDropTotal - 1}, ramp agrees with the reference"
  IO.println "ConvNeXt-T:"
  for si in [0:4] do
    for j in [0:(#[3,3,9,3] : Array Nat)[si]!] do
      let i := cnxBlockIdx si j
      IO.println s!"  stage {si} block {j} → ramp {i}: keep {convnextVerified.dropKeeps[i]!}"
  IO.println s!"✓ {cnxDropSites} sites, denominator {cnxDropTotal - 1}, ramp agrees with the reference"
  IO.println "ViT-Tiny:"
  for i in [0:vitDropTotal] do
    IO.println s!"  block {i}: sites {vitSiteIdx i 0}/{vitSiteIdx i 1} (attn/mlp) → keep \
{vitVerified.dropKeeps[vitSiteIdx i 0]!}"
  IO.println s!"✓ {vitDropSites} sites at {vitDropTotal} keeps, denominator {vitDropTotal - 1}, \
two INDEPENDENT masks per block sharing one keep"
