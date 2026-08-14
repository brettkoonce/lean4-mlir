import LeanMlir
import LeanMlir.Proofs.Codegen.ResNet34RenderB

/-! # The OPTIMIZER STAGE alone, as a runnable module — `scripts/opt_step_tie.py`'s input

`planning/verified_optimizer_parity.md` §5: *the reference and the verified path share a data
pipeline by construction and share an optimizer by nobody's construction.* `tests/vjp_oracle` diffs
them at the **gradient**; until now nothing diffed them at the **update**. This file emits the half
that gate needs — one optimizer step, as a function of `(θ, g, m, v, G)` — for each variant.

## ⚠⚠ Why the gradients are function ARGUMENTS here

A rendered train step computes its own gradients from `%x`, so diffing `θ'` against the reference
would confound optimizer differences with gradient differences — and the gradient half is exactly
what `vjp_oracle` already covers. Taking `g` as an input is what isolates the update.

## ⚠⚠ WHAT MAKES THIS A GATE RATHER THAN A SECOND IMPLEMENTATION

The body is `Proofs.StableHLO.optAllParams`, **the same call `resnet50TrainStepFaithfulB` makes** —
same clip hoist, same all-reduce placement, same accumulator handling, same `optOne`. Nothing about
the optimizer is spelled here; this file supplies a signature, a parameter list and a `return`.

That is not a stylistic preference. §5's own lesson one level down: *a gate on a copy is not a gate
on the thing copied.* `optAllParams` was inline in `ResNet50RenderB.lean` until 2026-08-14 and was
factored out **for this**, byte-identically (all 176 artifacts unchanged).

## The parameter set, and why these three shapes

| name | shape | what it exercises |
|---|---|---|
| `p0` | `[4,3,2,2]` | rank-4, DECAYS (`r34WdDecays`: rank ≥ 2), takes the real trust ratio |
| `p1` | `[5]` | rank-1 — the `no_weight_decay` group: `%wdz`, and D2's `trust = 1` skip |
| `p2` | `[6,4]` | rank-2, decays. ⚠ A SECOND decaying parameter is not redundant |

⚠ **Three, not one, and never fewer than two decaying ones.** The clip's whole semantic content is
that ONE scalar is shared across every parameter (`Proofs.clipFactor_shared` against
`Proofs.lambScale_not_shared` — LAMB's genuinely per-tensor ratio, three ops away in the same
graph). A single-parameter fixture cannot tell the two apart: with one leaf the global norm IS the
per-tensor norm, and a per-parameter clip would pass every check this gate makes.

⚠ The rank-1 row is load-bearing for the same reason in the other direction: under `wx` it is the
only one taking `%wdz`/`%lzero`, so a fixture of rank-≥2 parameters alone would gate D2 vacuously.

    lake build opt-step-fixtures && .lake/build/bin/opt-step-fixtures
    scripts/opt_step_tie.py
-/

open Proofs.StableHLO

/-- The fixture's parameters: `(name, shape)`. See the header for why there are three. -/
private def fixtureParams : List (String × List Nat) :=
  [("p0", [4,3,2,2]), ("p1", [5]), ("p2", [6,4])]

/-- One optimizer-only module. `optAllParams` supplies the body; everything here is interface.

    ⚠ The gradient of `pI` arrives as `%dpI`, a function argument, and the parameter/moment/
    accumulator slots keep the `%pI` / `%pIm` / `%pIv` / `%pIa` spelling `optOne` emits against —
    those names are a CONTRACT with the optimizer, not a local choice. -/
private def optStepModule (fname : String) (opt : R34Opt)
    (wdExclude : Bool := false) (gradClip : Bool := false) (clipNorm : Float := 1.0)
    -- ⚠ `wdStr` reaches ONLY the constant block: the per-parameter ops name `%wd` as an operand, so
    -- overriding the decay changes one `stablehlo.constant` and nothing else. That is exactly the
    -- property `optWdStr`'s docstring claims, and this fixture is where it gets measured.
    (wdStr : String := "") : String :=
  let accOn := match opt with | .adamwAccum _ => true | .lambAccum _ => true | _ => false
  let ps : List PGrad := fixtureParams.map (fun (n, ds) => ⟨n, s!"%d{n}", ds⟩)
  let (body, thetaN, mN, vN, aN) := (optAllParams opt 1 1 ps wdExclude gradClip clipNorm).run' 0
  let tys := fixtureParams.map (fun (_, ds) => ty ds)
  let sig := String.intercalate ", " (
    fixtureParams.map (fun (n, ds) => s!"%{n}: {ty ds}") ++
    fixtureParams.map (fun (n, ds) => s!"%{n}m: {ty ds}") ++
    fixtureParams.map (fun (n, ds) => s!"%{n}v: {ty ds}") ++
    (if accOn then fixtureParams.map (fun (n, ds) => s!"%{n}a: {ty ds}") else []) ++
    ["%lr: tensor<f32>", "%bc1: tensor<f32>", "%bc2: tensor<f32>"] ++
    (if accOn then ["%aup: tensor<f32>", "%akeep: tensor<f32>"] else []) ++
    fixtureParams.map (fun (n, ds) => s!"%d{n}: {ty ds}"))
  let outTys := String.intercalate ", " (tys ++ tys ++ tys ++ (if accOn then tys else []))
  let retVals := String.intercalate ", " (thetaN ++ mN ++ vN ++ aN)
  "module @m {\n" ++
  s!"  func.func @{fname}({sig}) -> ({outTys}) " ++ "{\n" ++
  -- ⚠ Same constant prelude the real render emits, from the same three functions — a hand-written
  -- `%b1 = 0.9` here would make the gate measure this file's transcription of the recipe.
  optConstsB opt wdStr ++ wdzConst wdExclude ++ clipZeroConst gradClip ++ body ++
  s!"    return {retVals} : {outTys}\n" ++
  "  }\n}\n"

/-- The variants the gate runs. ⚠⚠ **EVERY ROW IS CHOSEN TO MATCH A REFERENCE THAT ACTUALLY
    SHIPS**, because `scripts/opt_step_tie.py` EXECUTES the generated reference's own optimizer
    lines rather than re-implementing them — so a variant with no corresponding generated file
    could only be gated against a transcription, which is not a gate (`grad_tie.py`'s standing rule:
    *the reference must be the GENERATED reference*).

    | slug | reference | how |
    |---|---|---|
    | `lamb` | `generated_resnet50_imagenet.py` | LAMB, no mask, no clip, no accum — exact |
    | `lambwxclip` | `generated_resnet50_imagenet_a2accum.py` | the same code at `_K = 1`, which IS the no-accumulation case (`grads = _gsum/1`) |
    | `lambacc4wxclip` | `…a2accum.py` | `_K = 4`, the config's own `GRAD_ACCUM` — exact |
    | `lambacc8wxclip` | `…a2accum.py` | `_K = 8`, RSB-A3's k. The extracted region is k-agnostic |

    ⭐ `lambacc{4,8}wxclip` are the rows D1 exists for: they are the only ones where the baked
    threshold is ARITHMETIC (`k·C`, `clipNormStr`) rather than a transcribed literal. `lambwxclip`
    is their `k = 1` control — same clip, no scaling — so a failure in the accumulating rows and not
    in that one localises to the accumulation arithmetic rather than to the clip.

    ⚠ **`.adamw` IS NOT COVERED, and that is a real gap rather than an oversight.** No generated
    reference bakes R50's AdamW constants (`%eps = 1e-8`, `%wd = 1e-4`) — the Adam-family references
    in the tree are EfficientNet's and MNv2's TF-RMSProp recipes at `EPS = 1e-3`/`1.0`. Gating
    `.adamw` here would mean writing the reference by hand, which is the thing this file refuses to
    do. ▶ It wants a config that generates it, not a transcription. -/
private def variants : List (String × R34Opt × Bool × Bool × String) :=
  [ -- (slug, optimizer, wdExclude, gradClip, wdStr)
    ("lamb",           .lamb,        false, false, "")   -- the trust ratio alone
  , ("lambwxclip",     .lamb,        true,  true,  "")   -- + D2 mask + D1 clip, no accumulation
  , ("lambacc4wxclip", .lambAccum 4, true,  true,  "")   -- ⭐ D1 × accumulation at the config's k
  , ("lambacc8wxclip", .lambAccum 8, true,  true,  "")   -- ⭐⭐ RSB-A3's actual composition
    -- ⭐ RSB-**A1**'s decay, 0.01 against A3's 0.02 — the `wdStr` knob, and the row that makes it a
    -- MEASURED knob rather than a threaded string. Its reference is `…_a1.py`, which bakes
    -- `WD = 0.010000`, so a `wdStr` that failed to reach the constant block would show up as a
    -- wrong θ' here and nowhere else. ⚠ The slug carries `wd001` because `wdVariantMark` puts it
    -- there: two renders differing only in a baked constant must not share a path.
  , ("lambacc8wxclipwd001", .lambAccum 8, true, true, "0.01")
    -- ⭐⭐ **`.adamw`, AND IT IS ONLY EXPRESSIBLE BECAUSE OF THE ROW ABOVE.** The gap this file
    -- recorded was that no generated reference baked R50's AdamW constants. `adam-probe` bakes
    -- `eps = 1e-8` (the render's) and `wd = 0.02` (NOT the render's 1e-4), so before `wdStr` made
    -- the decay a render parameter, closing it would have meant hand-writing a reference — the one
    -- thing this file refuses to do. ▶ A knob paying for a gate two features later.
  , ("adamwxclipwd002", .adamw, true, true, "0.02") ]

def main : IO Unit := do
  IO.println "── optimizer-step fixtures (planning/verified_optimizer_parity.md §5) ──"
  for (slug, opt, wx, clip, wd) in variants do
    let fname := s!"opt_step_{slug}"
    let m := optStepModule fname opt wx clip 1.0 wd
    let path := s!".lake/build/{fname}.mlir"
    IO.FS.writeFile path m
    IO.println s!"  wrote {path} ({(m.splitOn "\n").length} lines)"
  IO.println s!"✓ {variants.length} variants"
